"""Rewritten program: high-precision ODE integrator for Lotka-Volterra."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system with high accuracy and few function calls.
    Uses adaptive Dormand-Prince 5(4) method with tight tolerances.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 280
    
    # Dormand-Prince 5(4) coefficients
    # a[i][j] for i=2..7, j=1..i-1 (stage i uses stages 1..i-1)
    # c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    # b4 (4th order) = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    # b5 (5th order) = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, -1/40]
    
    a = np.zeros((7, 7))
    a[1, 0] = 1/5
    a[2, 0] = 3/40
    a[2, 1] = 9/40
    a[3, 0] = 44/45
    a[3, 1] = -56/15
    a[3, 2] = 32/9
    a[4, 0] = 5
    a[4, 1] = -20
    a[4, 2] = 120/11
    a[4, 3] = -81/11
    a[5, 0] = -71/5760
    a[5, 1] = 71/16695
    a[5, 2] = -71/1920
    a[5, 3] = 172537/330000
    a[5, 4] = -2205/1088
    a[6, 0] = 71/57600
    a[6, 1] = -71/16695
    a[6, 2] = 71/1920
    a[6, 3] = -172537/330000
    a[6, 4] = 2205/1088
    a[6, 5] = 1/40
    
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    b4 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    b5 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, -1/40])
    
    rtol = 1e-12
    atol = 1e-14
    
    t = t0
    h = (t1 - t0) / 50
    y = np.array(y0, dtype=float)
    
    max_steps = 100000
    step = 0
    
    while t < t1 - 1e-15 and step < max_steps:
        if t + h > t1:
            h = t1 - t
        
        # Compute stages
        k = np.zeros((7, len(y)))
        k[0] = f(t, y)
        
        for i in range(1, 7):
            yi = y + h * sum(a[i, j] * k[j] for j in range(i))
            k[i] = f(t + c[i] * h, yi)
        
        # 5th order solution
        y5 = y + h * (b5 @ k)
        # 4th order solution
        y4 = y + h * (b4 @ k)
        
        # Error estimate
        err_vec = y5 - y4
        scale = atol + rtol * np.maximum(np.abs(y4), np.abs(y5))
        err_norm = np.linalg.norm(err_vec / scale, ord=np.inf)
        
        if err_norm <= 1.0 or h < 1e-12:
            t = t + h
            y = y5
            if err_norm > 0:
                h = min(h * 0.9 * err_norm**(-0.2), 2.0 * h)
            else:
                h = 2.0 * h
        else:
            h = max(h * 0.9 * err_norm**(-0.25), h * 1e-6)
        
        step += 1
        
        if time.time() > deadline:
            break
    
    if t < t1 - 1e-15:
        h = t1 - t
        k = np.zeros((7, len(y)))
        k[0] = f(t, y)
        for i in range(1, 7):
            yi = y + h * sum(a[i, j] * k[j] for j in range(i))
            k[i] = f(t + c[i] * h, yi)
        y = y + h * (b5 @ k)
    
    return y
# EVOLVE-BLOCK-END
