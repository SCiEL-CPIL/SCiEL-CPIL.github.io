"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    start_time = time.time()
    deadline = 295.0  # safety margin before 300s limit

    # Primary: high-order adaptive DOP853 with very tight tolerances
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: slightly looser DOP853
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-10,
            atol=1e-12,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: Radau (implicit, handles stiffness)
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            (t0, t1),
            y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: fixed-step RK4
    h = (t1 - t0) / 2000
    t, y = t0, y0.copy()
    for _ in range(2000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if time.time() - start_time > deadline:
            break
    return y
# EVOLVE-BLOCK-END
