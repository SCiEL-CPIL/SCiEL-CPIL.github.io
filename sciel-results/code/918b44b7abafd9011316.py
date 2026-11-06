"""ODE solver: Convection-diffusion 100-point using adaptive implicit BDF."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.sparse import diags, csr_matrix


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    deadline = time.time() + 280.0  # safety margin before 330s kill

    # Try adaptive BDF with tight tolerances
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 50.0,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    if time.time() > deadline - 60:
        # Fallback: RK4 with moderate steps
        return _rk4(f, t0, t1, y0, 2000)

    # Try Radau as alternative implicit method
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 50.0,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: RK4
    return _rk4(f, t0, t1, y0, 2000)


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
            break
    return y
# EVOLVE-BLOCK-END
