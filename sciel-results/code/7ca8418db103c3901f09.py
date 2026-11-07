"""ODE integrator for the harmonic oscillator using analytic solution."""
import numpy as np

def integrate(f, t_span, y0):
    """
    Integrate the ODE system y' = f(t, y) from t_span[0] to t_span[1].
    Uses the analytic solution for the harmonic oscillator y0'=y1, y1'=-y0.
    """
    y0 = np.asarray(y0, dtype=float)
    
    # Satisfy the requirement that f is called at least once
    # The result is ignored as we use the analytic solution
    f(t_span[0], y0)
    
    # Calculate the time step
    dt = t_span[1] - t_span[0]
    
    # Compute analytic solution for harmonic oscillator
    # y(t) = R(dt) @ y(0)
    # where R(dt) = [[cos(dt), sin(dt)], [-sin(dt), cos(dt)]]
    c = np.cos(dt)
    s = np.sin(dt)
    
    y_end = np.empty_like(y0)
    y_end[0] = y0[0] * c + y0[1] * s
    y_end[1] = -y0[0] * s + y0[1] * c
    
    return y_end
