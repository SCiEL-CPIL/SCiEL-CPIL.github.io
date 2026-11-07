"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) using a fixed-step Radau IIA
    collocation method with a small number of steps, solving each step's
    nonlinear system with Newton iteration.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Use a moderate number of steps for accuracy and efficiency
    # With 20 steps, h=0.5, which should be fine for Radau IIA (3rd order)
    # Actually, let's use more steps for better accuracy
    # Let's try 50 steps: h = 0.2
    num_steps = 50
    h = (t1 - t0) / num_steps
    
    # Radau IIA 3-stage collocation nodes and weights
    # For s=3 stages:
    c = np.array([0.2388834764831845, 0.4595222244569569, 1.0])
    A = np.array([
        [0.1287797589758319, 0.0738264532196684, 0.0362937642036842],
        [0.1724428643806137, 0.2017003430460506, 0.0853767925733357],
        [0.2388834764831845, 0.4595222244569569, 1.0]
    ])
    b = np.array([0.2388834764831845, 0.4595222244569569, 1.0])
    
    # Precompute I - h*A for the linear system in each Newton iteration
    # The Radau IIA method solves:
    # y_{n+1} = y_n + h * sum_j b_j f(t_n + c_j h, Y_j)
    # Y_i = y_n + h * sum_j A_ij f(t_n + c_j h, Y_j)
    #
    # This is a system of s*n equations in s*n unknowns (the stage values Y_i)
    # For each step, we solve this nonlinear system with Newton's method.
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    for step in range(num_steps):
        t_n = t0 + step * h
        
        # Initial guess for stage values: use y_n for all stages
        Y = np.tile(y, (3, 1))  # shape (3, n)
        
        # Newton iteration for the stage values
        for newton_iter in range(15):
            # Evaluate f at stage points
            F = np.array([f(t_n + c[i] * h, Y[i]) for i in range(3)])
            
            # Residual: Y_i - y_n - h * sum_j A_ij F_j
            residual = Y - y[None, :] - h * (A @ F)
            
            # Jacobian of the residual with respect to Y
            # d(residual_i)/dY_k = delta_ik I - h * A_ik * J_f
            # where J_f is the Jacobian of f at each stage
            # For a 2D system with 3 stages, this is a 6x6 matrix
            
            # Compute finite-difference Jacobians
            J_f = np.zeros((3, n, n))
            eps = 1e-7
            for i in range(3):
                for j in range(n):
                    e = np.zeros(n)
                    e[j] = eps * max(1.0, abs(Y[i, j]))
                    F_plus = f(t_n + c[i] * h, Y[i] + e)
                    F_minus = f(t_n + c[i] * h, Y[i] - e)
                    J_f[i, :, j] = (F_plus - F_minus) / (2 * e[j])
            
            # Build the full 6x6 Jacobian (for n=2, s=3: 6x6)
            size = 3 * n
            J_full = np.zeros((size, size))
            for i in range(3):
                for k in range(3):
                    if i == k:
                        J_full[i*n:(i+1)*n, k*n:(k+1)*n] = np.eye(n)
                    J_full[i*n:(i+1)*n, k*n:(k+1)*n] -= h * A[i, k] * J_f[k]
            
            # Solve the Newton system
            res_flat = residual.flatten()
            try:
                delta = np.linalg.solve(J_full, -res_flat)
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse
                delta = np.linalg.lstsq(J_full, -res_flat, rcond=None)[0]
            
            Y += delta.reshape(3, n)
            
            if np.linalg.norm(delta) < 1e-12 * (1 + np.linalg.norm(Y)):
                break
        
        # Update y using the collocation formula
        F = np.array([f(t_n + c[i] * h, Y[i]) for i in range(3)])
        y = y + h * (b @ F)
        y = np.array(y, dtype=float)
        
        # Check for non-finite values
        if not np.all(np.isfinite(y)):
            break
        
        # Check time budget
        if time.time() > deadline:
            break
    
    return y
# EVOLVE-BLOCK-END
