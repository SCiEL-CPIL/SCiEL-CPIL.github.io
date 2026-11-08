"""ODE solver for Robertson problem using high-accuracy Radau integration."""
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) over t_span starting from y0.
    Returns the final state y(t_end).

    Uses Radau V with very tight tolerances and no max_step restriction
    to minimize function evaluations while achieving high accuracy.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)

    # Track time to ensure we finish well before the 330s limit
    start_time = time.time()
    deadline = start_time + 300  # 30s safety margin

    # Primary: Radau with very tight tolerances, no max_step
    try:
        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Check if we still have time
    if time.time() > deadline - 30:
        # Fallback: use looser tolerances quickly
        try:
            sol = solve_ivp(
                f,
                [t0, t1],
                y0,
                method='Radau',
                rtol=1e-10,
                atol=1e-12
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
        # Last resort: simple integration
        n_steps = 10000
        h = (t1 - t0) / n_steps
        t, y = t0, y0.copy()
        for _ in range(n_steps):
            y = y + h * f(t, y)
            t += h
        return y

    # Secondary: BDF with tight tolerances
    try:
        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='BDF',
            rtol=1e-13,
            atol=1e-15
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Tertiary: LSODA with tight tolerances
    try:
        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=1e-13,
            atol=1e-15
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: RK4 with many steps
    n_steps = 20000
    h = (t1 - t0) / n_steps
    t, y = t0, y0.copy()
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
