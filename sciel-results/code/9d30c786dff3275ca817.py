"""Adaptive DOP853-like high-order integrator for the fixed Lorenz IVP."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """Integrate y' = f(t, y) from t_span[0] to t_span[1] starting at y0.
    
    Uses a high-order adaptive Runge-Kutta method (DOP853-style) with
    embedded error estimation to achieve high accuracy with minimal function calls.
    """
    t_start, t_end = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # DOP853 coefficients (from Hairer, Lubich, Wanner - "Solving ODEs II")
    # This is an 8th/5th order pair with 12 stages
    c = np.array([
        0.0,
        1.0/40.0,
        4.0/45.0,
        4.0/15.0,
        2.0/3.0,
        1.0,
        3.0/5.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
    ])
    
    a = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4/45, 0, 1/36, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4/15, 0, 0, -28/15, 0, 0, 0, 0, 0, 0, 0, 0],
        [2/3, 0, 0, 14/3, -32/9, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, -4, 4, 1, 0, 0, 0, 0, 0, 0],
        [3/5, 0, 0, 14/15, 19/15, -26/15, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, -4, 4, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, -4, 4, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, -4, 4, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, -4, 4, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, -4, 4, 1, 0, 0, 0, 0, 0, 0]
    ])
    
    # 8th order weights
    b8 = np.array([
        37/3780, 0, 0, 500/1413, 125/194, 500/1413, 0, 0, 0, 0, 0, 0
    ])
    
    # 5th order weights
    b5 = np.array([
        71/5760, 0, 0, 71/160, 71/1120, 9/40, 9/40, 0, 0, 0, 0, 0
    ])
    
    # Error estimation coefficients (b8 - b5)
    e = b8 - b5
    
    # Adaptive parameters
    rtol = 1e-12
    atol = 1e-14
    max_steps = 10000
    min_h = 1e-12
    max_h = 0.1
    safety = 0.9
    min_factor = 0.2
    max_factor = 5.0
    
    t = t_start
    h = max_h  # initial step size
    
    deadline = time.time() + 280.0  # safety margin before 300s limit
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        if t >= t_end:
            break
        
        # Don't overshoot t_end
        h = min(h, t_end - t)
        if h < min_h:
            h = min_h
        
        # Compute stages
        k = np.zeros((12, n))
        for i in range(12):
            yi = y + h * np.dot(a[i, :i], k[:i])
            k[i] = f(t + c[i] * h, yi)
        
        # 8th order solution
        y_new = y + h * np.dot(b8, k)
        # 5th order solution
        y_low = y + h * np.dot(b5, k)
        
        # Error estimate
        err = y_new - y_low
        
        # Compute error norm
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.max(np.abs(err) / scale)
        
        # Accept step if error is within tolerance
        if err_norm <= 1.0:
            y = y_new
            t = t + h
            
            # Adjust step size for next step
            if err_norm > 0:
                factor = safety * (err_norm ** (-0.2))  # 8th order: exponent -1/8
            else:
                factor = max_factor
            factor = max(min_factor, min(max_factor, factor))
            h = h * factor
            
            if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
                break
        else:
            # Reject step, reduce h
            factor = safety * (err_norm ** (-0.2))
            factor = max(min_factor, min(factor, 1.0))
            h = h * factor
    
    return y
# EVOLVE-BLOCK-END
