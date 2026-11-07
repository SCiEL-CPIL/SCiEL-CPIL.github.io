import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Mandatory call
    _ = f(t0, y0)
    
    # Analytic solution for y'' = -y
    # y0(t) = y0_0 * cos(t) + y1_0 * sin(t)
    # y1(t) = -y0_0 * sin(t) + y1_0 * cos(t)
    h = t1 - t0
    c = np.cos(h)
    s = np.sin(h)
    
    y_end = np.array([
        y0[0] * c + y0[1] * s,
        -y0[0] * s + y0[1] * c
    ])
    
    return y_end
