"""ODE integrator for the Lotka-Volterra system."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate as _si


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    deadline = time.time() + 280.0

    # Strategy: Use DOP853 with tighter tolerances than the reference
    # to drive the relative error down to the 1e-14 floor.
    # The reference uses rtol=1e-11, which likely limits accuracy to ~1e-11.
    # By using rtol=1e-13, we aim for rel ~ 1e-14, gaining ~3 points in -log10(rel).
    # The cost in calls (N) is logarithmic, so the net score should improve.
    
    try:
        sol = _si.solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method="DOP853",
            rtol=1e-13,
            atol=1e-15,
        )
        if sol.success and sol.t[-1] >= t1 - 1e-12:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)) and y_end.shape == y0_arr.shape:
                return y_end
    except Exception:
        pass

    # Fallback if the tighter tolerance failed or was too slow (unlikely for this ODE)
    if time.time() > deadline:
        return y0_arr

    try:
        sol = _si.solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
        )
        if sol.success and sol.t[-1] >= t1 - 1e-12:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)) and y_end.shape == y0_arr.shape:
                return y_end
    except Exception:
        pass

    if time.time() > deadline:
        return y0_arr

    # Final fallback: RK4 with many fixed steps
    h = (t1 - t0) / 20000
    t, y = t0, y0_arr.copy()
    n_steps = 20000
    for _ in range(n_steps):
        if time.time() > deadline - 5.0:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break

    return y
# EVOLVE-BLOCK-END
