"""High-precision ODE integrator for the Lotka-Volterra system using DOP853."""

import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    # Solve the IVP using DOP853 with very tight tolerances to match
    # the reference solution computed with rtol=1e-11, atol=1e-13.
    # Using even tighter tolerances to minimize relative error.
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-14,
        atol=1e-16,
        dense_output=False
    )
    
    # Return the state at t_end
    y_end = sol.y[:, -1]
    
    # Ensure finite values and correct shape
    y_end = np.asarray(y_end, dtype=float)
    if not np.all(np.isfinite(y_end)):
        # Fallback to a very large step RK4 if something went wrong
        n = len(y0)
        y = np.asarray(y0, dtype=float).copy()
        t0, t1 = t_span
        h = (t1 - t0) / 100000
        for _ in range(100000):
            k1 = f(t0, y)
            k2 = f(t0 + h/2, y + h/2 * k1)
            k3 = f(t0 + h/2, y + h/2 * k2)
            k4 = f(t0 + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t0 = t0 + h
        y_end = y
    
    return y_end
