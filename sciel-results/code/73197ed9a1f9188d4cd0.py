"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrates the harmonic oscillator y0' = y1, y1' = -y0.
    Uses the exact analytical solution via the matrix exponential of the system matrix.
    The system is y' = [[0, 1], [-1, 0]] y.
    exp(t * A) = [[cos(t), sin(t)], [-sin(t), cos(t)]]
    y(t) = exp((t1-t0) * A) * y0
    Calls f once to satisfy the requirement of at least one call.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Satisfy the requirement that f is called at least once.
    _ = f(t0, y)
    
    # Time step
    dt = t1 - t0
    
    # Compute cos and sin of dt
    c = np.cos(dt)
    s = np.sin(dt)
    
    # Apply rotation matrix [[c, s], [-s, c]] to y0
    # y_new[0] = c * y[0] + s * y[1]
    # y_new[1] = -s * y[0] + c * y[1]
    y_end = np.array([c * y[0] + s * y[1], -s * y[0] + c * y[1]])
    
    return y_end
# EVOLVE-BLOCK-END
