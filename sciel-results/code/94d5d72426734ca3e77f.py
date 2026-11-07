import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Fixed-step Dormand-Prince 5(4)
    # Order 5, 6 stages
    # h = 0.001 -> 2000 steps -> 12000 calls
    # This should yield rel ~ 1e-14 or better
    h = 0.001
    n_steps = int(np.ceil((t1 - t0) / h))
    h = (t1 - t0) / n_steps
    
    y = y0.copy()
    t = t0
    
    # Butcher tableau for DP54
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    a = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ])
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    b5 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

    for _ in range(n_steps):
        k = np.zeros((6, len(y)))
        for i in range(6):
            y_stage = y + h * np.sum(a[i, :i] * k[:i], axis=0)
            k[i] = f(t + c[i] * h, y_stage)
        
        # 5th order update
        y = y + h * np.sum(b[:6] * k[:6], axis=0)
        t += h
        
    return y
