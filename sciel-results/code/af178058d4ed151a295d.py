import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Max calls allowed is 2,000,000.
    # RK4 uses 4 calls per step.
    # Max steps = 500,000.
    # The score function is approx 3.5 * log10(N) + const, meaning
    # increasing N (steps) increases the score. We use the maximum
    # allowed steps to minimize truncation error.
    steps = 500000
    
    h = (t1 - t0) / steps
    t = t0
    
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h

    return y
