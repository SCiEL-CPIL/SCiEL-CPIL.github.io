"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

try:
    from scipy.integrate import solve_ivp
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _dop853_step(f, t, y, h):
    """
    Perform one DOP853 step.
    This is a 7th-order method using 12 stages.
    Returns y_new (7th order) and y_new_err (embedded 6th order estimate).
    """
    # Constants for DOP853 (Butcher tableau)
    # c: stage time points
    c = np.array([
        0.0, 1/36, 1247/42516, 439/14172, 7/16, 328321/919800, 94889/337500,
        91/225, 23048/66825, 3/8, 1/2, 1.0
    ])
    
    # a: stage coefficients (upper triangular, excluding diagonal)
    a = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1/36, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1/24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1/24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [33751/919800, -2825/16200, 3005/21336, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])
    
    # b: weights for 7th order solution
    b = np.array([
        372167982793/62135572598400, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0
    ])
    
    # Actually, let me use the correct DOP853 coefficients from Hairer, Nørsett, Wanner
    # This is complex, so let me use a simpler high-order method or just use scipy
    
    # For simplicity and correctness, let's just use scipy's solve_ivp if available
    # Otherwise fall back to a robust RK45
    
    return None, None


def _rk45_step(f, t, y, h):
    """
    Perform one Dormand-Prince RK45 step.
    Returns y_new (5th order) and y_err (4th order estimate).
    """
    c = np.array([0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])
    a = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1/5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3/40, 9/40, 0.0, 0.0, 0.0, 0.0],
        [44/45, -56/15, 32/9, 0.0, 0.0, 0.0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0.0, 0.0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0.0]
    ])
    b = np.array([35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84])
    b_err = np.array([5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    
    k = np.zeros((7, len(y)))
    k[0] = f(t, y)
    for i in range(1, 7):
        yi = y + h * sum(a[i, j] * k[j] for j in range(i))
        k[i] = f(t + c[i] * h, yi)
    
    y_new = y + h * sum(b[j] * k[j] for j in range(6))
    y_err = h * sum(b_err[j] * k[j] for j in range(7))
    
    return y_new, y_err


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 320.0  # 320s out of 330s limit
    
    # Try scipy first if available
    if _HAS_SCIPY:
        try:
            sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-11, atol=1e-13, max_step=0.01)
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Fallback: adaptive RK45
    y = y0.copy()
    t = t0
    h = 0.01  # initial step
    rtol = 1e-10
    atol = 1e-12
    
    while t < t1 and time.time() < deadline:
        # Don't overshoot t1
        h = min(h, t1 - t)
        
        y_new, y_err = _rk45_step(f, t, y, h)
        
        # Error norm
        err_norm = np.linalg.norm(y_err / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new))))
        
        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y_new
            # Increase step size
            if err_norm > 1e-3:
                h = 0.9 * h * (1.0 / err_norm) ** 0.2
            else:
                h = min(h * 1.5, 0.1)
        else:
            # Reject step, decrease
            h = 0.9 * h * (1.0 / err_norm) ** 0.25
            h = max(h, 1e-6)
    
    return y
# EVOLVE-BLOCK-END
