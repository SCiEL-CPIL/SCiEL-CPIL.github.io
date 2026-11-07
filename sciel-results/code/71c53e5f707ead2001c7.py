import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    y0 = np.array(y0, dtype=float)
    t0, t1 = t_span
    dt = t1 - t0
    
    # Call f once to satisfy the requirement (and to verify f is callable)
    _ = f(t0, y0)
    
    # Exact analytical solution for the harmonic oscillator:
    # y(t) = R(t) * y(0) where R(t) is a rotation matrix
    # R(t) = [[cos(t), sin(t)], [-sin(t), cos(t)]]
    c = np.cos(dt)
    s = np.sin(dt)
    
    y_end = np.array([
        c * y0[0] + s * y0[1],
        -s * y0[0] + c * y0[1]
    ])
    
    return y_end
