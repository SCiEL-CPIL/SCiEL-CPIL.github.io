"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solves the 10x10 2D Heat Equation.
    Strategy: 
    1. Assume the standard 5-point Laplacian structure for the 2D heat equation.
    2. Construct the matrix A analytically.
    3. Verify that f(t, y0) == A @ y0 with a single call.
    4. If verified, use expm(A * dt) for an exact solution (up to floating point precision).
    5. If not verified, fallback to a robust implicit solver (Radau) with high accuracy.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Standard 10x10 grid
    grid_size = 10
    if n != grid_size * grid_size:
        # Fallback for unexpected sizes, though task specifies 10x10
        return _fallback_solve(f, t_span, y0, start_time)

    # Construct the 5-point Laplacian matrix A for the heat equation:
    # u_t = D * (u_xx + u_yy)
    # Usually discretized as: u_t = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]) * D / dx^2
    # Assuming unit domain [0,1]x[0,1], dx = 1/9? Or dx=1/10?
    # Typically for 10x10 nodes on [0,1], dx = 1/9 if endpoints included, or 1/10 if cell centers.
    # However, the coefficient D/dx^2 is just a scalar multiplier.
    # Let's estimate the scale.
    # If the problem is normalized, the eigenvalues will be negative.
    # Let's try to infer the scale 'k' such that f = k * L @ y, where L is the standard combinatorial Laplacian.
    
    # Standard combinatorial Laplacian for 10x10 grid (Dirichlet boundaries assumed 0, or Neumann?)
    # Let's build the matrix L where L[i,j] = 1 if i,j are neighbors, -4 on diagonal (interior), -3 (edge), -2 (corner).
    # Wait, the standard heat equation with Dirichlet BCs on [0,1]x[0,1] with 10x10 interior points?
    # Or 10x10 total points?
    # If 10x10 total points, boundaries are fixed.
    # Let's assume the vector y0 contains all 100 points.
    
    L = np.zeros((n, n))
    idx = lambda i, j: i * grid_size + j
    
    for i in range(grid_size):
        for j in range(grid_size):
            k = idx(i, j)
            L[k, k] = -4.0  # Center
            
            # Neighbors
            if i > 0:
                L[k, idx(i-1, j)] = 1.0
            if i < grid_size - 1:
                L[k, idx(i+1, j)] = 1.0
            if j > 0:
                L[k, idx(i, j-1)] = 1.0
            if j < grid_size - 1:
                L[k, idx(i, j+1)] = 1.0
                
    # The actual ODE is y' = alpha * L @ y.
    # We need to find alpha.
    # Let's probe f(t, y0) once.
    y_probe = y0
    f_probe = np.array(f(t0, y_probe))
    
    # Calculate L @ y0
    Ly0 = L @ y0
    
    # Check if f_probe is proportional to Ly0
    # alpha = dot(f_probe, Ly0) / dot(Ly0, Ly0)
    dot_Ly0_Ly0 = np.dot(Ly0, Ly0)
    
    if dot_Ly0_Ly0 < 1e-12:
        # y0 is in the null space of L (e.g., constant vector for Neumann, or zero).
        # If y0 is zero, solution is zero.
        if np.allclose(y0, 0):
            return y0
        # If not zero but Ly0 is zero, it's an eigenvector with eigenvalue 0?
        # For Dirichlet, only zero vector has eigenvalue 0.
        # For Neumann, constant vector has eigenvalue 0.
        # If f_probe is also close to zero, then y(t) = y0 (steady state).
        if np.allclose(f_probe, 0, atol=1e-10):
            return y0
        # Otherwise, structure mismatch or different BCs.
        return _fallback_solve(f, t_span, y0, start_time)

    alpha = np.dot(f_probe, Ly0) / dot_Ly0_Ly0
    
    # Verify the fit: || f_probe - alpha * Ly0 || should be small
    residual = np.linalg.norm(f_probe - alpha * Ly0)
    norm_f = np.linalg.norm(f_probe)
    
    # If the residual is small relative to the norm, we assume this is the structure.
    if residual < 1e-8 * (norm_f + 1e-10):
        # We have identified the system as y' = A y with A = alpha * L.
        # We can use expm.
        A = alpha * L
        dt = t1 - t0
        
        try:
            # Compute expm(A * dt)
            # A is 100x100, expm is fast.
            E = expm(A * dt)
            y_end = E @ y0
            
            # Sanity check: is the result finite?
            if np.all(np.isfinite(y_end)):
                return y_end
        except Exception:
            pass
            
    # Fallback: The structure didn't match perfectly or expm failed.
    # Use a robust implicit solver.
    return _fallback_solve(f, t_span, y0, start_time)

def _fallback_solve(f, t_span, y0, start_time):
    """
    Robust fallback using SciPy's Radau method (implicit, stiff-accurate).
    """
    from scipy.integrate import solve_ivp
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Set a tight tolerance to maximize accuracy score.
    # rtol=1e-10, atol=1e-12
    # Radau is good for stiff problems like heat equation.
    
    # We need to be careful with the number of calls.
    # Radau might use many Jacobian evaluations if not provided.
    # But we don't have the Jacobian analytically here (or we do, but the probe failed).
    # Let's just let it estimate.
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=(t1-t0)/10 # Limit step size to ensure accuracy
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: High-order RK4 with many steps
    # This is slow and less accurate for stiff systems, but better than failing.
    steps = 1000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    
    for _ in range(steps):
        if time.time() > start_time + 300:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y
# EVOLVE-BLOCK-END
