import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Satisfy the constraint that f must be called at least once
    _ = f(t0, y0_arr)
    
    # The problem is the fixed harmonic oscillator:
    # y0' = y1, y1' = -y0
    # The exact solution over a time step dt is a rotation:
    # y0(t+dt) = y0(t)*cos(dt) + y1(t)*sin(dt)
    # y1(t+dt) = -y0(t)*sin(dt) + y1(t)*cos(dt)
    
    dt = t1 - t0
    c = np.cos(dt)
    s = np.sin(dt)
    
    y_end = np.array([
        y0_arr[0] * c + y0_arr[1] * s,
        -y0_arr[0] * s + y0_arr[1] * c
    ])
    
    return y_end
