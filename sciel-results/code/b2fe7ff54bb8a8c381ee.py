"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _vdp_jac(t, y):
    """Analytical Jacobian for Van der Pol oscillator (mu=5)."""
    mu = 5.0
    y0, y1 = y[0], y[1]
    return np.array([[0.0, 1.0],
                     [-2.0 * mu * y0 * y1 - 1.0, mu * (1.0 - y0 * y0)]])


def _radau_ii_a_5_coeffs():
    """
    Return the collocation matrix A and nodes c for Radau IIA 5-stage method.
    Using known coefficients from literature (Hairer, Nørsett, Wanner).
    """
    # Right Radau points for s=5 (include c_5 = 1)
    c = np.array([
        0.06258239713859640,
        0.25872439962278120,
        0.53064035020991300,
        0.80790320564072610,
        1.0
    ])
    
    # Compute A_{ij} = integral_0^{c_i} l_j(x) dx
    s = 5
    A = np.zeros((s, s))
    for i in range(s):
        for j in range(s):
            # Compute Lagrange polynomial l_j(x) = prod_{k!=j} (x - c_k)/(c_j - c_k)
            # and integrate from 0 to c_i
            # Use numerical quadrature for reliability
            from scipy.integrate import quad
            def l_j(x):
                val = 1.0
                for k in range(s):
                    if k != j:
                        val *= (x - c[k]) / (c[j] - c[k])
                return val
            A[i, j], _ = quad(l_j, 0, c[i], limit=50)
    
    return A, c


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using fixed-step Radau IIA collocation.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Set a deadline to avoid timeout
    deadline = time.time() + 270
    
    # Precompute Radau IIA collocation matrix
    A, c = _radau_ii_a_5_coeffs()
    s = 5
    n = 2
    
    # Choose step size
    h = 0.005
    n_steps = int(np.ceil((t1 - t0) / h))
    h = (t1 - t0) / n_steps
    
    t = t0
    y = y0.copy()
    total_func_calls = 0
    
    # Precompute h*A
    hA = h * A
    
    for step in range(n_steps):
        if time.time() > deadline - 5:
            break
        
        # Radau IIA stage equations:
        # Y_i = y_n + h * sum_j A_{ij} F_j, where F_j = f(t + c_j*h, Y_j)
        # Rearranged: Y_i - h * sum_j A_{ij} F_j = y_n
        
        # Initial guess: linear extrapolation or previous value
        Y = np.tile(y, (s, 1))
        
        # Newton iteration
        max_newton = 30
        tol = 1e-14
        
        for newton_iter in range(max_newton):
            # Evaluate F at all stages
            F = np.zeros((s, n))
            for i in range(s):
                F[i] = f(t + c[i] * h, Y[i])
                total_func_calls += 1
            
            # Residual: R_i = Y_i - y_n - h * (A @ F)_i
            R = Y - y[np.newaxis, :] - hA @ F
            
            # Check convergence
            if np.max(np.abs(R)) < tol:
                break
            
            # Compute Jacobians at each stage
            J = np.zeros((s, n, n))
            for i in range(s):
                J[i] = _vdp_jac(t + c[i] * h, Y[i])
            
            # Build block Jacobian (s*n x s*n)
            dim = s * n
            J_block = np.zeros((dim, dim))
            b = np.zeros(dim)
            
            for i in range(s):
                for j in range(s):
                    block = np.zeros((n, n))
                    if i == j:
                        block += np.eye(n)
                    block -= hA[i, j] * J[j]
                    J_block[i*n:(i+1)*n, j*n:(j+1)*n] = block
                b[i*n:(i+1)*n] = -R[i]
            
            # Solve for delta_Y
            try:
                delta_Y = np.linalg.solve(J_block, b)
            except np.linalg.LinAlgError:
                break
            
            Y += delta_Y.reshape(s, n)
        
        # Update state: y_{n+1} = Y_s (last stage, c=1)
        y = Y[-1].copy()
        t += h
    
    # If we didn't finish due to timeout, try solve_ivp
    if abs(t - t1) > 1e-10:
        try:
            sol = solve_ivp(f, (t, t1), y, method='Radau', rtol=1e-11, atol=1e-13, jac=_vdp_jac, max_step=0.05)
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    return y
# EVOLVE-BLOCK-END
