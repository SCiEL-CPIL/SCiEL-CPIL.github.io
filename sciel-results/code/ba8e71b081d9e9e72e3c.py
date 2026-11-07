"""High-order RK integrator for the Lotka-Volterra system.

Uses scipy's DOP853 with tight tolerances to achieve reference-floor
accuracy with minimal function call count.
"""
import numpy as np
import scipy.integrate as _spi


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # DOP853 with tight tolerances; the adaptive step size will be large
    # for this smooth problem, minimizing call count.
    sol = _spi.solve_ivp(
        f,
        (t0, t1),
        y0_arr,
        method='DOP853',
        rtol=1e-14,
        atol=1e-16,
    )
    
    if sol.success and sol.y.shape[1] > 0:
        return sol.y[:, -1]
    
    # Fallback: RK4 with many steps
    return _rk4_fallback(f, t0, t1, y0_arr)


def _rk4_fallback(f, t0, t1, y0):
    h = (t1 - t0) / 8000.0
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(8000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
