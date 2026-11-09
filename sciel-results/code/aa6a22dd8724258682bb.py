import numpy as np
import scipy.integrate

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Use Radau with very tight tolerances to match the reference solution closely
    # This minimizes relative error while keeping call count reasonable
    sol = scipy.integrate.solve_ivp(
        f,
        t_span,
        y0,
        method='Radau',
        rtol=1e-12,
        atol=1e-14,
        max_step=0.01
    )
    if not sol.success:
        # Fallback with slightly looser tolerances if the strict one fails
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13
        )
    return sol.y[:, -1]
