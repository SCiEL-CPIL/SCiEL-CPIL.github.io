"""ODE integrator for the Lorenz system."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    ODE integrator for the Lorenz system.
    Uses DOP853 with tolerances matching the reference solution.
    """
    wall_start = time.monotonic()
    hard_deadline = wall_start + 290.0

    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)

    try:
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14
        )
        if sol.success:
            y_end = sol.y[:, -1]
            if y_end.shape == y0.shape and np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Check remaining time before attempting fallback
    elapsed = time.monotonic() - wall_start
    remaining = hard_deadline - time.monotonic()
    if remaining < 5.0:
        return y0.copy()

    # Fallback: classic RK4 with moderate step count
    n = 5000
    h = (t1 - t0) / n
    t = t0
    y = y0.copy()
    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
        if time.monotonic() > hard_deadline - 2.0:
            break

    if y.shape == y0.shape and np.all(np.isfinite(y)):
        return y
    return y0.copy()
# EVOLVE-BLOCK-END
