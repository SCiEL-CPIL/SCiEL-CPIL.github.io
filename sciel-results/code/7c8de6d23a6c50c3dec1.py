"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm1m

def _estimate_jacobian(f, t, y, h=None):
    """Estimate the Jacobian matrix of f(t, y) with respect to y."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if h is None:
        h = 1e-7 * np.maximum(1.0, np.abs(y))
    
    J = np.zeros((n, n), dtype=float)
    y0 = y.copy()
    f0 = np.asarray(f(t, y0), dtype=float)
    
    for j in range(n):
        y_pert = y0.copy()
        y_pert[j] += h[j]
        f_pert = np.asarray(f(t, y_pert), dtype=float)
        J[:, j] = (f_pert - f0) / h[j]
        
    return J

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float)
    n = len(y)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 250.0  # 300s limit, leave 50s margin
    
    # Estimate A at t0
    A0 = _estimate_jacobian(f, t0, y)
    
    # Check if system is approximately time-invariant.
    # We use a quick explicit RK4 to get an approximate y(t1) to evaluate A(t1).
    # If RK4 fails (blows up), we assume time-dependent and use more steps.
    
    def _quick_rk4(f, t0, t1, y0, steps=10):
        h = (t1 - t0) / steps
        t, y = t0, np.array(y0, dtype=float)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            if not np.all(np.isfinite(y)):
                return None
        return y

    y_approx_t1 = _quick_rk4(f, t0, t1, y)
    
    use_single_step = False
    if y_approx_t1 is not None:
        A1 = _estimate_jacobian(f, t1, y_approx_t1)
        diff_A = np.linalg.norm(A0 - A1)
        norm_A = np.linalg.norm(A0)
        rel_diff = diff_A / (norm_A + 1e-10)
        
        if np.isfinite(rel_diff) and rel_diff < 0.1:
            use_single_step = True
            
    if use_single_step:
        K = 1
    else:
        K = 10
        
    # Perform integration
    h = (t1 - t0) / K
    t = t0
    y_curr = y.copy()
    
    for k in range(K):
        if k == 0:
            A_curr = A0
        else:
            A_curr = _estimate_jacobian(f, t, y_curr)
            
        # Compute matrix exponential
        M = expm1m(A_curr * h)
        y_curr = M @ y_curr
        t += h
        
        # Deadline check
        if time.time() > deadline:
            break
            
    return y_curr
# EVOLVE-BLOCK-END
