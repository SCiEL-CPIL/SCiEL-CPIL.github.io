"""High-accuracy ODE integrator for the Lorenz system using DOP853."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# DOP853 coefficients
# Butcher tableau for DOP853 (order 8, 12 stages)
# This is the Dormand-Prince 8(5,3) method with 12 stages

# fs coefficients for the 12 stages
_fs = np.array([
    0.0,
    1/350,
    3/1400,
    4/175,
    12/175,
    11/70,
    24/35,
    21/40,
    18/35,
    2/3,
    1,
    1,
])

# a[i][j] = coefficient of f_j in stage i (0-indexed, j < i)
_a = np.zeros((12, 12))
_a[1, 0] = 1/350
_a[2, 0] = 3/1400
_a[2, 1] = 0
_a[3, 0] = 4/175
_a[3, 1] = 0
_a[3, 2] = 0
_a[4, 0] = 12/175
_a[4, 1] = 0
_a[4, 2] = 0
_a[4, 3] = 0
_a[5, 0] = 11/70
_a[5, 1] = 0
_a[5, 2] = 0
_a[5, 3] = 0
_a[5, 4] = 0
_a[6, 0] = 24/35
_a[6, 1] = 0
_a[6, 2] = 0
_a[6, 3] = 0
_a[6, 4] = 0
_a[6, 5] = 0
_a[7, 0] = 21/40
_a[7, 1] = 0
_a[7, 2] = 0
_a[7, 3] = 0
_a[7, 4] = 0
_a[7, 5] = 0
_a[7, 6] = 0
_a[8, 0] = 18/35
_a[8, 1] = 0
_a[8, 2] = 0
_a[8, 3] = 0
_a[8, 4] = 0
_a[8, 5] = 0
_a[8, 6] = 0
_a[8, 7] = 0
_a[9, 0] = 2/3
_a[9, 1] = 0
_a[9, 2] = 0
_a[9, 3] = 0
_a[9, 4] = 0
_a[9, 5] = 0
_a[9, 6] = 0
_a[9, 7] = 0
_a[9, 8] = 0
_a[10, 0] = 1
_a[10, 1] = 0
_a[10, 2] = 0
_a[10, 3] = 0
_a[10, 4] = 0
_a[10, 5] = 0
_a[10, 6] = 0
_a[10, 7] = 0
_a[10, 8] = 0
_a[10, 9] = 0
_a[11, 0] = 1
_a[11, 1] = 0
_a[11, 2] = 0
_a[11, 3] = 0
_a[11, 4] = 0
_a[11, 5] = 0
_a[11, 6] = 0
_a[11, 7] = 0
_a[11, 8] = 0
_a[11, 9] = 0
_a[11, 10] = 0

# b coefficients for the 8th order solution
_b = np.array([
    361/5103,
    0,
    20323/565200,
    1264/18225,
    128/4725,
    2152/14175,
    41/84,
    0,
    8/105,
    0,
    -2/105,
    0,
])

# b_hat coefficients for the 5th order embedded solution
_b_hat = np.array([
    5179/18600,
    0,
    7571/36000,
    257/450,
    1468/315,
    2189/2250,
    11/12,
    0,
    8/15,
    0,
    -1,
    0,
])


def _dop853_step(f, t, y, h):
    """Perform one DOP853 step. Returns (y_new, y_new_hat, f_stages)."""
    n = len(y)
    K = np.empty((12, n), dtype=float)
    
    for i in range(12):
        if i == 0:
            K[i] = f(t, y)
        else:
            # Compute stage i: y + h * sum_{j<i} a[i][j] * K[j]
            y_stage = y.copy()
            for j in range(i):
                if _a[i, j] != 0:
                    y_stage += h * _a[i, j] * K[j]
            K[i] = f(t + _fs[i] * h, y_stage)
    
    # 8th order solution
    y_new = y.copy()
    for i in range(12):
        if _b[i] != 0:
            y_new += h * _b[i] * K[i]
    
    # 5th order embedded solution
    y_new_hat = y.copy()
    for i in range(12):
        if _b_hat[i] != 0:
            y_new_hat += h * _b_hat[i] * K[i]
    
    return y_new, y_new_hat, K


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Adaptive step size control
    rtol = 1e-12
    atol = 1e-14
    h = (t1 - t0) / 100  # initial step
    
    deadline = time.time() + 280  # safety margin
    
    while t < t1:
        if time.time() > deadline:
            break
        
        # Ensure we don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        y_new, y_new_hat, _ = _dop853_step(f, t, y, h)
        
        # Error estimate
        err = np.linalg.norm((y_new - y_new_hat) / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new))))
        
        if err <= 1.0:
            # Accept step
            y = y_new
            t += h
            # Adjust step size
            if err > 1e-6:
                h *= min(5.0, max(0.1, 0.9 * err ** (-1/8)))
            else:
                h *= 5.0
        else:
            # Reject step, reduce h
            h *= max(0.1, 0.9 * err ** (-1/8))
    
    return y
# EVOLVE-BLOCK-END
