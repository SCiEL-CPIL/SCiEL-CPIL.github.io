import numpy as np
import scipy.integrate

def integrate(f, t_span, y0):
    y0 = np.asarray(y0, dtype=float)
    # Use tighter tolerances to minimize rel error, aiming for the 1e-14 floor
    # This may increase calls, but the gain in -log10(rel) likely outweighs the cost
    sol = scipy.integrate.solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-14,
        atol=1e-16,
        dense_output=False
    )
    if sol.success and sol.y.shape[1] > 0:
        y_end = sol.y[:, -1]
        if np.all(np.isfinite(y_end)):
            return y_end
    return y0
