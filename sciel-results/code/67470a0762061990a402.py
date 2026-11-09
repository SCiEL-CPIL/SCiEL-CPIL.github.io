# EVOLVE-BLOCK-START
import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float).copy()
    
    # Number of steps
    n_steps = 10000
    h = (t1 - t0) / n_steps
    t = t0
    
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        
    return y
# EVOLVE-BLOCK-END
