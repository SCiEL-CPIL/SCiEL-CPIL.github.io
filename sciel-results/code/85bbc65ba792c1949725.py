"""High-order ODE integrator for the harmonic oscillator using SciPy's DOP853."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] starting at y0.
    Uses scipy.integrate.solve_ivp with DOP853 method to match the reference solution
    closely while minimizing function calls via adaptive step size control.
    """
    sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-11, atol=1e-13)
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
