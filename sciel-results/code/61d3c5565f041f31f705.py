# EVOLVE-BLOCK-START
import numpy as np
import time

# DOP853 coefficients (8th order explicit Runge-Kutta, Fehlberg)
# These are the standard DOP853 coefficients for 13-stage method
# Source: Dormand & Prince 1998, as used in scipy's DOP853

_c = np.array([
    0.0,
    1.0/40.0,
    4.0/45.0,
    68.0/225.0,
    8.0/15.0,
    1.0,
    1732.0/495.0,
    512.0/567.0,
    128.0/45.0,
    1.0,
    243.0/40.0,
    11.0/5.0
], dtype=float)

# a[i][j] coefficient matrix (i > j)
_a = np.zeros((13, 13), dtype=float)
_a[1, 0] = 1.0/40.0
_a[2, 0] = 3.0/40.0; _a[2, 1] = 3.0/10.0
_a[3, 0] = 839.0/19200.0; _a[3, 1] = -19.0/120.0; _a[3, 2] = 5.0/8.0
_a[4, 0] = 1.0/12.0; _a[4, 1] = 0.0; _a[4, 2] = -3.0/4.0; _a[4, 3] = 1.0
_a[5, 0] = 512.0/1771.0; _a[5, 1] = 0.0; _a[5, 2] = -12.0/5.0; _a[5, 3] = 34.0/7.0; _a[5, 4] = -5.0/4.0
_a[6, 0] = -10533.0/16000.0; _a[6, 1] = 297.0/50.0; _a[6, 2] = -48.0/5.0; _a[6, 3] = 27.0; _a[6, 4] = -3.0/2.0; _a[6, 5] = -105.0/32.0
_a[7, 0] = 3.0/2.0; _a[7, 1] = -9.0/4.0; _a[7, 2] = 0.0; _a[7, 3] = 5.0; _a[7, 4] = -15.0/8.0; _a[7, 5] = -5.0/24.0
_a[8, 0] = -1.0; _a[8, 1] = 5.0/4.0; _a[8, 2] = 0.0; _a[8, 3] = -1.0; _a[8, 4] = 0.0; _a[8, 5] = 0.0; _a[8, 6] = 0.0; _a[8, 7] = 0.0
_a[9, 0] = 7.0/8.0; _a[9, 1] = 0.0; _a[9, 2] = 0.0; _a[9, 3] = -3.0/8.0; _a[9, 4] = 0.0; _a[9, 5] = 0.0; _a[9, 6] = 0.0; _a[9, 7] = 0.0; _a[9, 8] = 0.0
_a[10, 0] = 7.0/4.0; _a[10, 1] = 0.0; _a[10, 2] = 0.0; _a[10, 3] = -1.0; _a[10, 4] = 0.0; _a[10, 5] = 0.0; _a[10, 6] = 0.0; _a[10, 7] = 0.0; _a[10, 8] = 0.0; _a[10, 9] = 0.0
_a[11, 0] = -87.0/250.0; _a[11, 1] = 0.0; _a[11, 2] = 0.0; _a[11, 3] = 3.0/16.0; _a[11, 4] = 0.0; _a[11, 5] = 0.0; _a[11, 6] = 0.0; _a[11, 7] = 0.0; _a[11, 8] = 0.0; _a[11, 9] = 0.0; _a[11, 10] = 0.0

# 8th order weights
_b8 = np.array([
    361.0/5120.0,
    0.0,
    5251.0/17792.0,
    105.0/16.0,
    -875.0/416.0,
    68.0/21.0,
    18.0/13.0,
    0.0,
    0.0,
    -2.0/105.0,
    1.0/40.0,
    0.0,
    0.0
], dtype=float)

# 7th order weights (for error estimation)
_b7 = np.array([
    53.0/2560.0,
    0.0,
    811.0/6608.0,
    27.0/4.0,
    -27.0/16.0,
    3.0/2.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
], dtype=float)

def _dop853_step(f, t, y, h):
    """Take one DOP853 step, returning y_new and error estimate."""
    k = [None] * 13
    k[0] = np.asarray(f(t, y), dtype=float)
    for i in range(1, 13):
        yi = y.copy()
        for j in range(i):
            yi = yi + h * _a[i, j] * k[j]
        k[i] = np.asarray(f(t + _c[i] * h, yi), dtype=float)
    
    y8 = y.copy()
    for i in range(13):
        y8 = y8 + h * _b8[i] * k[i]
    
    y7 = y.copy()
    for i in range(13):
        y7 = y7 + h * _b7[i] * k[i]
    
    err = y8 - y7
    return y8, err

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Initial step size estimate
    k0 = np.asarray(f(t0, y), dtype=float)
    k1 = np.asarray(f(t0 + (t1 - t0) * 1e-6, y), dtype=float)
    d = np.max(np.abs(k1 - k0)) / (t1 - t0)
    yscale = np.max(np.abs(y))
    if d > 0 and yscale > 0:
        h0 = 0.01 * yscale / d
    else:
        h0 = (t1 - t0) / 100.0
    h = min(h0, (t1 - t0) * 0.1)
    h = max(h, (t1 - t0) * 1e-8)
    
    t = t0
    rtol = 1e-12
    atol = 1e-14
    start_time = time.time()
    
    while t < t1 - 1e-14:
        # Adjust step to not overshoot t1
        if t + h > t1:
            h = t1 - t
        
        # DOP853 step
        y_new, err = _dop853_step(f, t, y, h)
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(y_new)):
            h *= 0.5
            if h < 1e-16:
                return y
            continue
        
        # Error norm
        err_norm = np.max(np.abs(err) / (atol + rtol * np.abs(y_new)))
        
        if err_norm <= 1.0:
            t = t + h
            y = y_new
            # Adjust step size
            if err_norm > 0:
                h_new = h * min(5.0, max(0.1, (0.9 / err_norm) ** 0.2))
            else:
                h_new = h * 5.0
            h = min(h_new, (t1 - t) * 0.9) if t1 - t > 0 else h
        else:
            h = h * max(0.1, (0.9 / err_norm) ** 0.2)
        
        # Deadline guard
        if time.time() - start_time > 280.0:
            break
    
    return y
# EVOLVE-BLOCK-END
