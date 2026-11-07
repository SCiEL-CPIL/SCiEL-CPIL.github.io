import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n_steps = 20000
    h = (t1 - t0) / n_steps
    
    for _ in range(n_steps):
        k1 = f(t0, y)
        k2 = f(t0 + h/2, y + h/2 * k1)
        k3 = f(t0 + h/2, y + h/2 * k2)
        k4 = f(t0 + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t0 += h
        
    return y
