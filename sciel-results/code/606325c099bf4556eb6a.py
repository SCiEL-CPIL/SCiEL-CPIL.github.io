"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] using an adaptive
    Dormand-Prince 5(4) method (DOPRI5) with tight tolerances.
    
    Returns y(t_end).
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, 20s safety margin
    
    # Dormand-Prince 5(4) Butcher tableau
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
    ])
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    b_star = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    
    # Tolerances
    rtol = 1e-10
    atol = 1e-12
    
    # Initial step size
    h = (t1 - t0) / 1000.0
    t = t0
    
    # Safety factor
    SAFETY = 0.9
    MIN_FACTOR = 0.1
    MAX_FACTOR = 10.0
    
    # First derivative
    k1 = f(t, y)
    
    # Initial error estimate for step size selection
    # Use a rough estimate
    err_norm = np.max(np.abs(k1) * h)
    if err_norm == 0:
        h = (t1 - t0) / 100.0
    else:
        scale = rtol * np.max(np.abs(y)) + atol
        h = 0.01 * scale / err_norm
        h = min(h, (t1 - t0) / 10.0)
        h = max(h, (t1 - t0) / 100000.0)
    
    # Recompute k1 with new h
    k1 = f(t, y)
    
    while t < t1:
        # Check deadline
        if time.time() > deadline:
            return y
        
        # Ensure we don't overshoot t1
        if t + h > t1:
            h = t1 - t
        if h <= 0:
            break
        
        # Dormand-Prince 5(4) step
        k1 = f(t, y)
        k2 = f(t + c[1]*h, y + h*(A[1,0]*k1))
        k3 = f(t + c[2]*h, y + h*(A[2,0]*k1 + A[2,1]*k2))
        k4 = f(t + c[3]*h, y + h*(A[3,0]*k1 + A[3,1]*k2 + A[3,2]*k3))
        k5 = f(t + c[4]*h, y + h*(A[4,0]*k1 + A[4,1]*k2 + A[4,2]*k3 + A[4,3]*k4))
        k6 = f(t + c[5]*h, y + h*(A[5,0]*k1 + A[5,1]*k2 + A[5,2]*k3 + A[5,3]*k4 + A[5,4]*k5))
        k7 = f(t + h, y + h*(A[6,0]*k1 + A[6,1]*k2 + A[6,2]*k3 + A[6,3]*k4 + A[6,4]*k5 + A[6,5]*k6))
        
        # 5th order solution
        y_new = y + h*(b[0]*k1 + b[2]*k3 + b[3]*k4 + b[4]*k5 + b[5]*k6)
        
        # 4th order solution for error estimation
        y_err = y + h*(b_star[0]*k1 + b_star[2]*k3 + b_star[3]*k4 + b_star[4]*k5 + b_star[5]*k6 + b_star[6]*k7)
        
        # Error estimate
        err = y_new - y_err
        err_norm = np.max(np.abs(err) / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new))))
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(y_new)):
            # Reduce step and retry
            h = h * 0.1
            continue
        
        if err_norm <= 1.0:
            # Accept step
            t = t + h
            y = y_new
            
            # Adjust step size
            if err_norm > 0:
                factor = SAFETY * err_norm ** (-1.0/5.0)
            else:
                factor = MAX_FACTOR
            factor = min(MAX_FACTOR, max(MIN_FACTOR, factor))
            h = h * factor
        else:
            # Reject step, reduce h
            factor = SAFETY * err_norm ** (-1.0/5.0)
            factor = max(MIN_FACTOR, factor)
            h = h * factor
    
    return y
# EVOLVE-BLOCK-END
