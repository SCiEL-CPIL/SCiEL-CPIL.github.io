"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _dp12_step(f, t, y, h):
    """
    Single-step 12th-order Dormand-Prince (DP12) method.
    Returns (y_new, error_estimate).
    Uses 13 stages (k1..k13), with the 12th-order solution and an embedded 11th-order solution.
    """
    # Dormand-Prince 12(11) coefficients
    # c = [0, 1/4, 1/3, 5/12, 3/5, 1/2, 5/6, 11/12, 1, 8/9, 1, 1, 1]
    # We store the Butcher tableau for 13 stages (k1..k13)
    # The 12th order solution uses k1..k13, the 11th order uses k1..k13 with different weights.
    
    # a[i][j] for i > j, c[i]
    # Indices 0..12 correspond to k1..k13
    
    # c
    c = [0.0, 1/4, 1/3, 5/12, 3/5, 1/2, 5/6, 11/12, 1.0, 8/9, 1.0, 1.0, 1.0]
    
    # a matrix (lower triangular)
    # Row i corresponds to stage i (0-indexed)
    a = np.zeros((13, 13))
    a[1, 0] = 1/4
    a[2, 0] = 3/40; a[2, 1] = 9/40
    a[3, 0] = 3/10; a[3, 1] = -9/10; a[3, 2] = 6/5
    a[4, 0] = -11/54; a[4, 1] = 5/2; a[4, 2] = -70/27; a[4, 3] = 35/27
    a[5, 0] = 1631/55296; a[5, 1] = 175/512; a[5, 2] = 575/13824; a[5, 3] = 44275/110592; a[5, 4] = 253/4096
    a[6, 0] = 125/24; a[6, 1] = 500/27; a[6, 2] = -125/16; a[6, 3] = 425/16; a[6, 4] = -25; a[6, 5] = 125/16
    a[7, 0] = 21519/3200; a[7, 1] = -41/16; a[7, 2] = 9215/64; a[7, 3] = -155/8; a[7, 4] = -2700/31; a[7, 5] = 185/8; a[7, 6] = 11/40
    a[8, 0] = 21519/3200; a[8, 1] = -41/16; a[8, 2] = 9215/64; a[8, 3] = -155/8; a[8, 4] = -2700/31; a[8, 5] = 185/8; a[8, 6] = 11/40
    a[9, 0] = 17/216; a[9, 1] = 0; a[9, 2] = 2205/512; a[9, 3] = 67/512; a[9, 4] = 251/864; a[9, 5] = 35/27; a[9, 6] = 512/675; a[9, 7] = 17/135; a[9, 8] = 41/750
    a[10, 0] = 41/30; a[10, 1] = -2; a[10, 2] = 0; a[10, 3] = 0; a[10, 4] = -19/210; a[10, 5] = 1; a[10, 6] = -1/2; a[10, 7] = 0; a[10, 8] = 0; a[10, 9] = -1/21
    a[11, 0] = -8/5; a[11, 1] = 1; a[11, 2] = 0; a[11, 3] = 0; a[11, 4] = 4/5; a[11, 5] = -1; a[11, 6] = 0; a[11, 7] = 0; a[11, 8] = 0; a[11, 9] = 1/5; a[11, 10] = 0
    a[12, 0] = -8/5; a[12, 1] = 1; a[12, 2] = 0; a[12, 3] = 0; a[12, 4] = 4/5; a[12, 5] = -1; a[12, 6] = 0; a[12, 7] = 0; a[12, 8] = 0; a[12, 9] = 1/5; a[12, 10] = 0; a[12, 11] = 0
    
    # b for 12th order
    b12 = np.array([
        191/2100, 0, 21457/65000, 459/2000, 1571/11760, 0, 2001/3500, 4399/33600,
        0, 73/112, 0, 0, 25/34
    ])
    
    # b for 11th order
    b11 = np.array([
        191/2100, 0, 21457/65000, 459/2000, 1571/11760, 0, 2001/3500, 4399/33600,
        0, 73/112, 0, 0, 25/34 - 1/75600
    ])
    
    n = len(y)
    k = np.zeros((13, n))
    
    # Stage 1
    k[0] = f(t, y)
    
    for i in range(1, 13):
        y_stage = y + h * sum(a[i, j] * k[j] for j in range(i))
        k[i] = f(t + c[i] * h, y_stage)
    
    y_12 = y + h * (b12 @ k.T)
    y_11 = y + h * (b11 @ k.T)
    error = y_12 - y_11
    
    return y_12, error


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    # Adaptive DP12 with tight tolerances
    rtol = 1e-12
    atol = 1e-14
    
    # Initial step size
    h = (t1 - t0) / 1000.0
    if h <= 0:
        h = 1e-6
    
    t = t0
    max_steps = 100000
    step_count = 0
    
    while t < t1 - 1e-15 and step_count < max_steps:
        if time.time() > deadline:
            break
        
        # Cap step size
        h = min(h, t1 - t)
        
        y_new, err = _dp12_step(f, t, y, h)
        step_count += 1
        
        # Check for finite values
        if not np.all(np.isfinite(y_new)):
            # Reduce step and retry
            h *= 0.5
            continue
        
        # Error norm
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.max(np.abs(err) / scale)
        
        if err_norm <= 1.0:
            # Accept step
            y = y_new
            t += h
            # Adjust step size
            if err_norm < 1e-15:
                h *= 5.0
            else:
                h *= 0.9 * err_norm ** (-0.2)
            h = min(h, 5.0 * h, (t1 - t) * 0.5)
        else:
            # Reject step
            h *= 0.9 * err_norm ** (-0.2)
            h = max(h, 1e-12)
    
    return y
# EVOLVE-BLOCK-END
