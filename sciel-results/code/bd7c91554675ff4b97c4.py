"""ODE integrator for the harmonic oscillator using SciPy's DOP853 solver."""
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the ODE over t_span using DOP853 with tight tolerances
    to match the reference solution closely.
    """
    y0 = np.asarray(y0, dtype=float)
    sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-11, atol=1e-13)
    return sol.y[:, -1]
