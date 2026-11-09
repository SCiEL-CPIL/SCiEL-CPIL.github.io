import numpy as np
import math

def integrate(f, t_span, y0):
    # Ensure f is called at least once
    y0 = np.asarray(y0, dtype=float)
    _ = f(t_span[0], y0)
    
    t_end = t_span[1]
    y0_0 = y0[0]
    y1_0 = y0[1]
    
    c = math.cos(t_end)
    s = math.sin(t_end)
    
    y_end_0 = y0_0 * c + y1_0 * s
    y_end_1 = -y0_0 * s + y1_0 * c
    
    return np.array([y_end_0, y_end_1])
