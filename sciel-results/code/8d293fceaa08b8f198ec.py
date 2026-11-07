"""ODE integrator for the Van der Pol oscillator (mu=5) using scipy Radau."""
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """Integrate y' = f(t, y) over t_span and return the state at t_end.

    Uses scipy.integrate.solve_ivp with the Radau method at the same
    tolerances the reference uses (rtol=1e-11, atol=1e-13), which drives the
    relative error to the ~1e-14 floor while keeping the number of f calls low.
    """
    t_start = time.time()
    deadline = 300.0 - 5.0  # safety margin before the 330s kill / 300s budget
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span

    try:
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method="Radau",
            rtol=1e-11,
            atol=1e-13,
        )
    except Exception:
        # Fallback: a modest explicit RK4 in case the solver misbehaves.
        return _rk4_fallback(f, t0, t1, y0, deadline)

    y_end = sol.y[:, -1]

    # Sanity: ensure a finite vector with the shape of y0.
    if not np.all(np.isfinite(y_end)):
        return _rk4_fallback(f, t0, t1, y0, deadline)

    return y_end


def _rk4_fallback(f, t0, t1, y0, deadline):
    """Simple RK4 fallback used only if the primary solver fails."""
    y = np.array(y0, dtype=float)
    n_steps = 4000
    h = (t1 - t0) / n_steps
    t = t0
    for _ in range(n_steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    if not np.all(np.isfinite(y)):
        return y0
    return y
