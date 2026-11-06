"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] using DOP853 adaptive step size.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    # DOP853 coefficients
    # This is a 12-stage, 8th-order method with 5th-order error estimate
    # Coefficients from DOPRI853 (Hairer, Norsett, Wanner, "Solving Ordinary Differential Equations II")
    
    c = np.array([
        0.0,
        1/40.0,
        4/45.0,
        4/9.0,
        5/12.0,
        88/105.0,
        193/210.0,
        6/5.0,
        1.0,
        3/2.0,
        1.0,
        1.0
    ])
    
    a = np.zeros((12, 12))
    a[1, 0] = 1/40.0
    a[2, 0] = 4/45.0
    a[2, 1] = -28/15.0
    a[3, 0] = 28/9.0
    a[3, 1] = -2.0
    a[3, 2] = 1.0
    a[4, 0] = 371/110.0
    a[4, 1] = -532/55.0
    a[4, 2] = 692/55.0
    a[4, 3] = -172/11.0
    a[5, 0] = 4184/1425.0
    a[5, 1] = -3712/475.0
    a[5, 2] = 18544/1425.0
    a[5, 3] = -3552/375.0
    a[5, 4] = 712/375.0
    a[6, 0] = 381/140.0
    a[6, 1] = -54/35.0
    a[6, 2] = 156/35.0
    a[6, 3] = -132/35.0
    a[6, 4] = 132/35.0
    a[6, 5] = -66/35.0
    a[7, 0] = 285/28.0
    a[7, 1] = -9.0
    a[7, 2] = 36.0
    a[7, 3] = -16.0
    a[7, 4] = 8.0
    a[7, 5] = -4.0
    a[7, 6] = 2.0
    a[8, 0] = 133/120.0
    a[8, 1] = -1.0
    a[8, 2] = 0.0
    a[8, 3] = 0.0
    a[8, 4] = 0.0
    a[8, 5] = 0.0
    a[8, 6] = 0.0
    a[8, 7] = 1.0
    a[9, 0] = 59/60.0
    a[9, 1] = 1.0
    a[9, 2] = 0.0
    a[9, 3] = 0.0
    a[9, 4] = 0.0
    a[9, 5] = 0.0
    a[9, 6] = 0.0
    a[9, 7] = -1.0
    a[9, 8] = 1.0
    a[10, 0] = 1.0
    a[10, 1] = 0.0
    a[10, 2] = 0.0
    a[10, 3] = 0.0
    a[10, 4] = 0.0
    a[10, 5] = 0.0
    a[10, 6] = 0.0
    a[10, 7] = 0.0
    a[10, 8] = 0.0
    a[10, 9] = 0.0
    a[11, 0] = 1.0
    a[11, 1] = 0.0
    a[11, 2] = 0.0
    a[11, 3] = 0.0
    a[11, 4] = 0.0
    a[11, 5] = 0.0
    a[11, 6] = 0.0
    a[11, 7] = 0.0
    a[11, 8] = 0.0
    a[11, 9] = 0.0
    
    # 8th order weights
    b = np.array([
        361/5120.0,
        0.0,
        561/20480.0,
        -135/1024.0,
        7107/156250.0,
        -13967/737280.0,
        1261/122880.0,
        1404209/15552000.0,
        2107/163840.0,
        21187/3317760.0,
        0.0,
        0.0
    ])
    
    # 5th order weights for error estimation
    b5 = np.array([
        71/57600.0,
        0.0,
        0.0,
        -7/160.0,
        71/1152.0,
        -71/240.0,
        17/60.0,
        -41/300.0,
        1.0,
        0.0,
        0.0,
        0.0
    ])
    
    # Initial step size
    f0 = f(t0, y)
    n = len(y)
    
    # Estimate initial step size
    d0 = np.linalg.norm(y, ord=np.inf) / n
    d1 = np.linalg.norm(f0, ord=np.inf) / n
    if d0 < 1e-5:
        d0 = 1e-5
    if d1 < 1e-5:
        d1 = 1e-5
    h = 1e-6
    if d1 / d0 < 1e-5:
        h = 0.01 * (t1 - t0)
    else:
        h = 0.01 * (d0 / d1)
    h = min(h, t1 - t0)
    
    t = t0
    max_steps = 100000
    min_h = 1e-12
    max_h = (t1 - t0) * 0.5
    
    # Safety factor
    safety = 0.9
    min_factor = 0.2
    max_factor = 5.0
    
    for step in range(max_steps):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Don't overshoot
        if t + h > t1:
            h = t1 - t
        
        if h < min_h:
            h = min_h
        
        # Compute stages
        k = np.zeros((12, n))
        for i in range(12):
            y_stage = y.copy()
            for j in range(i):
                y_stage = y_stage + h * a[i, j] * k[j]
            k[i] = f(t + c[i] * h, y_stage)
        
        # Compute 8th order solution
        y_new = y.copy()
        for i in range(12):
            y_new = y_new + h * b[i] * k[i]
        
        # Compute 5th order solution for error estimation
        y_err = y.copy()
        for i in range(12):
            y_err = y_err + h * b5[i] * k[i]
        
        # Error estimate
        err = np.linalg.norm(y_new - y_err, ord=np.inf) / n
        
        # Accept/reject step
        if err <= 1e-10:
            # Accept step
            t = t + h
            y = y_new
            
            # Compute next step size
            if err > 0:
                factor = safety * (err / 1e-10) ** (-1/8)
            else:
                factor = max_factor
            factor = max(min_factor, min(factor, max_factor))
            h = h * factor
            h = min(h, max_h)
        else:
            # Reject step, try smaller h
            if err > 0:
                factor = safety * (err / 1e-10) ** (-1/8)
            else:
                factor = 0.5
            factor = max(min_factor, min(factor, 1.0))
            h = h * factor
            h = max(h, min_h)
        
        # Check if we've reached the end
        if t >= t1 - 1e-15:
            break
    
    return y
# EVOLVE-BLOCK-END
