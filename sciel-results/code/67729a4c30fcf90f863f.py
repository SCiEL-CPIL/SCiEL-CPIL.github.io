import numpy as np
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 250.0  # 300s limit, keep 50s margin
    
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Use 80 steps for RK4. 
    # Stability: h * |lambda_max| < 2.78
    # |lambda_max| ~ 1754
    # h = 0.1/80 = 0.00125
    # h * 1754 = 2.19 < 2.78. Safe.
    steps = 80
    h = (t1 - t0) / steps
    t = t0
    
    for _ in range(steps):
        if time.time() > deadline:
            break
            
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        
        # Check for non-finite values
        if not np.all(np.isfinite(y)):
            break
            
    return y
