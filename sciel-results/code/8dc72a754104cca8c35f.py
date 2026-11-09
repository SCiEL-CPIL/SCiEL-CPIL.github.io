import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13
    )
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback
        return y0.copy()
