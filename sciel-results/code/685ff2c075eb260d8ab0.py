"""ODE integrator for the harmonic oscillator using DOP853 (SciPy solve_ivp)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) over t_span starting from y0.
    
    Uses DOP853 with tight tolerances to match the reference solver's accuracy.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
        dense_output=False,
    )
    
    if not sol.success:
        # Fallback: use RK4 with many steps
        y = _rk4(f, t0, t1, y0, 10000)
        return y
    
    return sol.y[:, -1]


def _rk4(f, t0, t1, y0, steps):
    """Classic RK4 integrator as fallback."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            return y
    return y
# EVOLVE-BLOCK-END
