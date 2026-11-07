import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    dt = t1 - t0
    
    # Choose step size
    # h = 1e-3 gives 2000 steps.
    # Calls = 2000 * 4 = 8000.
    # This is well within 2,000,000 limit.
    h = 1e-3
    
    n_steps = int(np.ceil(dt / h))
    h = dt / n_steps
    
    for i in range(n_steps):
        t = t0 + i * h
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
    return y
