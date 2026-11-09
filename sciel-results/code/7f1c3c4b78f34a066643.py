"""High-order integrator for the fixed Lotka-Volterra IVP with tight tolerances."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Primary: DOP853 with very tight tolerances to minimize relative error.
    Fallback: fixed-step RK4 with many steps if solve_ivp fails.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)

    start_time = time.time()
    deadline = 290.0  # safety margin before 330s kill

    # Primary strategy: DOP853 with tighter tolerances than reference
    # to ensure our result is as close to the true solution as possible,
    # minimizing rel error against the reference (which has its own discretization error).
    try:
        sol = solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method="DOP853",
            rtol=1e-14,
            atol=1e-16,
            max_step=np.inf,
        )

        if sol.success and sol.y.shape[1] >= 1:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end

    except Exception:
        pass

    # Check deadline before fallback
    if time.time() > deadline:
        # Return a reasonable estimate: use a quick RK4 with fewer steps
        steps = 1000
        h = (t1 - t0) / steps
        t = t0
        y = y0_arr.copy()
        for _ in range(steps):
            k1 = np.asarray(f(t, y), dtype=float)
            k2 = np.asarray(f(t + h / 2.0, y + h / 2.0 * k1), dtype=float)
            k3 = np.asarray(f(t + h / 2.0, y + h / 2.0 * k2), dtype=float)
            k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
        return y

    # Fallback: fixed-step RK4 with a large number of steps
    steps = 10000
    h = (t1 - t0) / steps
    t = t0
    y = y0_arr.copy()

    for i in range(steps):
        if time.time() > deadline:
            break

        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + h / 2.0, y + h / 2.0 * k1), dtype=float)
        k3 = np.asarray(f(t + h / 2.0, y + h / 2.0 * k2), dtype=float)
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float)

        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h

    return y
# EVOLVE-BLOCK-END
