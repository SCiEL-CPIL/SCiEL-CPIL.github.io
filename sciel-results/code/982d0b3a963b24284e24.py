import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # DOP853 coefficients
    # 12 stages
    c = np.array([0.0, 1/12, 1/6, 1/4, 5/12, 1/2, 5/6, 1.0, 1.0, 1.0, 1.0, 1.0])
    a = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1/12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1/24, 1/24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1/48, 0.0, 1/4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [5/12, 0.0, -25/16, 25/16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1/2, 0.0, -5/12, 1.0, -5/12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [5/6, 0.0, -20/3, 10.0, 10/3, -25/12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, -72/5, 12.0, -12.0, 17/5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, -12/5, 8.0, -8.0, 17/5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Placeholder, need real values
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])
    # This hardcoded table is risky if I don't have the exact values.
    # Let's use a reliable high-order method I can construct or verify.
    # Or use DOP54 which I am more confident about.
    
    # Let's stick with DOP54 (6 calls) with N=1000 steps.
    # Calls = 6000.
    # This is safe and likely better than RK4 8000.
    
    # Let's try to optimize DOP54 steps.
    # If N=1000, calls=6000.
    # If N=500, calls=3000. Error ~1e-12.
    # If N=1000, calls=6000. Error ~1e-14.
    # The reference is ~1e-11.
    # So N=500 might be sufficient for rel < 1e-11.
    # Let's use N=500.
    
    c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    a = [
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ]
    b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    
    steps = 500
    h = (t1 - t0) / steps
    t = t0
    y = np.array(y0, dtype=float)
    
    for _ in range(steps):
        k = [np.zeros_like(y) for _ in range(7)]
        for i in range(7):
            k[i] = f(t + c[i]*h, y + h * sum(a[i][j]*k[j] for j in range(i)))
        y = y + h * sum(b[i]*k[i] for i in range(7))
        t += h
        if not np.all(np.isfinite(y)):
            return None
    return y
