"""ODE solver: 15-dimensional stiff linear system using analytic matrix exponential."""
import numpy as np
import time
from scipy.linalg import expm, solve


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) on [t0, t1] with initial condition y0.
    
    Strategy:
    1. Detect if the system is linear with constant coefficients: y' = A*y + b
    2. If so, use the analytic solution via block matrix exponential for robustness
    3. If not, fall back to a robust numerical method.
    
    The score is -log10(relative error) - 0.5*log10(number of f calls), so we want
    both high accuracy AND few function evaluations.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    T = t1 - t0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 50s margin before 330s kill
    
    # Step 1: Probe the system to determine if it's linear with constant coefficients
    # Evaluate b = f(t0, 0)
    y_zero = np.zeros(n)
    b = np.array(f(t0, y_zero), dtype=float)
    
    # Evaluate A columns: A[:, j] = (f(t0, e_j) - b) / 1.0
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j), dtype=float) - b
    
    # Verify linearity and constant coefficients with a test point
    y_test = np.ones(n) * 0.5 + 0.1 * np.arange(n)
    f_test = np.array(f(t0, y_test), dtype=float)
    f_predicted = A @ y_test + b
    residual = np.linalg.norm(f_test - f_predicted)
    scale = max(1.0, np.linalg.norm(f_predicted))
    
    # Also check at a different time to verify time-independence
    t_mid = (t0 + t1) / 2
    f_mid = np.array(f(t_mid, y_test), dtype=float)
    f_mid_predicted = A @ y_test + b
    residual_time = np.linalg.norm(f_mid - f_mid_predicted)
    
    is_linear_constant = (residual < 1e-10 * scale) and (residual_time < 1e-10 * scale)
    
    if is_linear_constant:
        # Use analytic solution via block matrix exponential
        # y(T) = expm(T*A) @ y0 + (integral_0^T expm(s*A) ds) @ b
        #
        # We compute this efficiently by solving a linear system:
        # Let E = expm(T*A). Then integral_0^T expm(s*A) ds = A^{-1}(E - I) if A invertible.
        # More robustly, we solve [T*A, -I; I, 0] [E; I] = [0; 0] for the integral part.
        # Actually, the standard trick is:
        #   [T*A  -I] [E    0 ]   [0  0]
        #   [ I    0] [0    I] = [I  0]
        # This gives E = expm(T*A) and the bottom-right block is the integral.
        # But scipy's expm already handles this well. Let's just use the augmented matrix
        # but compute only what we need.
        
        # Build the (n+1) x (n+1) augmented matrix
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A
        M[:n, n] = b
        # Last row is all zeros
        
        # Initial condition for augmented system
        z0 = np.zeros(n + 1)
        z0[:n] = y0
        
        # Compute matrix exponential and apply
        EM = expm(T * M)
        z_final = EM @ z0
        y_final = z_final[:n]
        
        return y_final
    
    # Fallback: use a robust numerical method
    from scipy.integrate import solve_ivp
    
    # Try Radau first (good for stiff problems)
    sol = solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-12, atol=1e-14, max_step=0.01)
    
    if sol.success and sol.t[-1] >= t1 - 1e-15:
        return sol.y[:, -1]
    
    # Try BDF
    sol = solve_ivp(f, [t0, t1], y0, method='BDF', rtol=1e-12, atol=1e-14, max_step=0.01)
    
    if sol.success and sol.t[-1] >= t1 - 1e-15:
        return sol.y[:, -1]
    
    # Last resort: use explicit RK45 with small steps
    sol = solve_ivp(f, [t0, t1], y0, method='RK45', rtol=1e-12, atol=1e-14, max_step=0.001)
    
    if sol.success:
        return sol.y[:, -1]
    
    # Absolute fallback: manual RK4 with many steps
    steps = 10000
    h = T / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if time.time() > deadline:
            break
    
    return y
