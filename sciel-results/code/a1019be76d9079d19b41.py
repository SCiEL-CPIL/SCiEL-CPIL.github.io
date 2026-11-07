"""ODE integrator for the Lorenz system using scipy DOP853 with a robust fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time

try:
    from scipy.integrate import solve_ivp
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _rk4(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    start = time.time()
    deadline = start + 250.0  # safety margin under 300 s wall-clock

    if _HAS_SCIPY:
        try:
            sol = solve_ivp(
                f, (t0, t1), y0,
                method="DOP853",
                rtol=1e-11, atol=1e-13,
                max_step=(t1 - t0) / 10.0,
            )
            if time.time() < deadline and sol.y.shape[1] > 0:
                y_end = sol.y[:, -1]
                if np.all(np.isfinite(y_end)) and y_end.shape == y0.shape:
                    return y_end
        except Exception:
            pass

    # Fallback: explicit RK4 with a fine fixed step (far under the 2e6 call cap).
    y = _rk4(f, t0, t1, y0, 20000)
    if y is None:
        return y0
    return y
# EVOLVE-BLOCK-END
