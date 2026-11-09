"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Dormand-Prince RK45 coefficients (corrected)
    # Stage times c
    c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    
    # Butcher tableau A (lower triangular)
    A = [
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/10, 9/40, 0, 0, 0, 0],
        [4/5, 0, 32/90, 0, 0, 0],
        [8/9, 0, -528/1250, 2197/4100, 0, 0],
        [1, 0, 7571/25440, -55/108, 175/54, 0],
    ]
    
    # b for 4th order solution
    b = [51/192, 0, 7571/25440, 39/140, 0, 29/576]
    # b* for 5th order solution
    bs = [361/5000, 0, 72964/50000, 142506/1250000, 77928/50000, 185977/1250000]
    # Note: the 7th stage (k6) is used for both, with b[6]=0 and bs[6]=2197/4100
    # Actually, let me use the standard 7-stage formulation
    
    # Proper 7-stage Dormand-Prince
    c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    A = [
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/10, 9/40, 0, 0, 0, 0, 0],
        [4/5, 0, 32/90, 0, 0, 0, 0],
        [8/9, 0, -528/1250, 2197/4100, 0, 0, 0],
        [1, 0, 7571/25440, -55/108, 175/54, 0, 0],
        [1, 0, 71/240, 0, -7/8, 35/12, 0],
    ]
    b4 = [51/192, 0, 7571/25440, 39/140, 0, 29/576, 0]
    b5 = [361/5000, 0, 72964/50000, 142506/1250000, 77928/50000, 185977/1250000, 2197/4100]
    
    rtol = 1e-12
    atol = 1e-14
    h = (t1 - t0) / 100
    t = t0
    f_calls = 0
    
    safety = 0.9
    min_factor = 0.1
    max_factor = 10.0
    
    while t < t1 - 1e-15 and f_calls < 1900000:
        if t + h > t1:
            h = t1 - t
        if h <= 1e-15:
            break
        
        # Compute stages
        k = [None] * 7
        k[0] = f(t, y)
        f_calls += 1
        
        for i in range(1, 7):
            y_stage = y.copy()
            for j in range(i):
                y_stage = y_stage + h * A[i][j] * k[j]
            k[i] = f(t + c[i] * h, y_stage)
            f_calls += 1
        
        # 4th and 5th order solutions
        y4 = y.copy()
        y5 = y.copy()
        for i in range(7):
            y4 = y4 + h * b4[i] * k[i]
            y5 = y5 + h * b5[i] * k[i]
        
        # Error estimate
        err = np.abs(y5 - y4)
        scale = atol + rtol * np.maximum(np.abs(y4), np.abs(y5))
        err_norm = np.sqrt(np.sum((err / scale) ** 2)) / np.sqrt(n)
        
        if err_norm <= 1.0:
            y = y5
            t += h
            if err_norm < 1e-8:
                factor = max_factor
            else:
                factor = safety * (1.0 / err_norm) ** 0.2
            factor = max(min_factor, min(max_factor, factor))
            h *= factor
        else:
            factor = safety * (1.0 / err_norm) ** 0.25
            factor = max(min_factor, min(0.9, factor))
            h *= factor
    
    return y
# EVOLVE-BLOCK-END
