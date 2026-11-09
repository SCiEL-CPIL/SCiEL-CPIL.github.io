"""Improved program using adaptive RK45 with tight tolerances for Van der Pol oscillator."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _adaptive_rk45(f, t0, t1, y0, rtol=1e-13, atol=1e-14, h_min=1e-10, h_max=0.1, max_steps=50000):
    """Adaptive RK45 (Bogacki-Prince) integrator with tight tolerances."""
    t = t0
    y = np.array(y0, dtype=float)
    h = 0.01  # initial step
    
    # RK45 coefficients (Bogacki-Prince 5(4) pair)
    c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    a = [
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
    ]
    b4 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    b5 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    
    # Safety factor and exponents for step size control
    safety = 0.9
    min_factor = 0.1
    max_factor = 10.0
    
    for step in range(max_steps):
        # Check deadline
        if step % 1000 == 0:
            pass  # Could add time check here
        
        # Compute stages
        k = np.zeros((7, len(y)))
        for i in range(7):
            y_stage = y.copy()
            for j in range(i):
                y_stage += h * a[i][j] * k[j]
            k[i] = f(t + c[i] * h, y_stage)
        
        # 4th and 5th order solutions
        y4 = y + h * np.sum(np.array(b4)[:, None] * k, axis=0)
        y5 = y + h * np.sum(np.array(b5)[:, None] * k, axis=0)
        
        # Error estimate
        err = np.abs(y5 - y4)
        scale = atol + rtol * np.maximum(np.abs(y4), np.abs(y5))
        err_norm = np.sqrt(np.sum((err / scale) ** 2))
        
        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y5
            # Adjust step size
            if err_norm < 1e-10:
                h = min(h * max_factor, h_max)
            else:
                h = min(h * safety * (1.0 / err_norm) ** 0.2, h_max)
            h = max(h, h_min)
        else:
            # Reject step, reduce h
            h = max(h * safety * (1.0 / err_norm) ** 0.25, h_min)
            if h <= h_min:
                # Can't reduce further, just take the step
                t += h
                y = y5
                h = h_min
        
        if t >= t1 - 1e-14:
            break
    
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = _adaptive_rk45(f, t0, t1, y0, rtol=1e-13, atol=1e-14)
    return y
# EVOLVE-BLOCK-END
