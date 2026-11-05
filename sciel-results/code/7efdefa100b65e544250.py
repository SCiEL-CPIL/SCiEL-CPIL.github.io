import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """Integrate y' = f(t, y) over t_span and return y at t_end."""
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    n = y0_arr.size
    deadline = time.time() + 290.0

    # Primary approach: DOP853 with max_step constraint to ensure resolution
    # max_step forces at least 50 steps, resolving convection-diffusion features
    best_y = None
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 50.0,
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            best_y = sol.y[:, -1].copy()
    except Exception:
        pass

    # Polish: if we have time, run again with tighter tolerances and more steps
    if best_y is not None and time.time() < deadline - 40.0:
        try:
            sol2 = solve_ivp(
                f,
                t_span,
                y0_arr,
                method='DOP853',
                rtol=1e-13,
                atol=1e-15,
                max_step=(t1 - t0) / 100.0,
            )
            if sol2.success and np.all(np.isfinite(sol2.y[:, -1])):
                best_y = sol2.y[:, -1].copy()
        except Exception:
            pass

    if best_y is not None:
        return best_y

    # Fallback: RK45 with tight tolerances and step constraint
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='RK45',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 100.0,
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: fixed-step RK4
    remaining = deadline - time.time()
    max_steps = int(remaining * 5000)
    max_steps = max(max_steps, 1000)
    max_steps = min(max_steps, 2000000 // 4)

    h = (t1 - t0) / max_steps
    t = t0
    y = y0_arr.copy()

    for i in range(max_steps):
        if time.time() > deadline - 5.0:
            break
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h

    return y
