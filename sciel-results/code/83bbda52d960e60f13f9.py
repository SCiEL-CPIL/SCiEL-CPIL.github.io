"""High-order integrator for the fixed Lotka-Volterra IVP with tight tolerances."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Primary: DOP853 with extremely tight tolerances (rtol=1e-14, atol=1e-16)
    to achieve near-machine-precision relative error with minimal function calls.
    Fallback: fixed-step RK4 with many steps if solve_ivp fails.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)

    start_time = time.time()

    # Primary strategy: DOP853 adaptive 8th-order Runge-Kutta
    # Using tighter tolerances than the reference (rtol=1e-11) ensures our error
    # is dominated by the reference's error or machine precision, minimizing rel.
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

    # Fallback: fixed-step RK4 with a large number of steps
    # For the LV system on [0, 5], the dynamics are smooth and non-stiff.
    # 10000 steps (h=0.0005) gives RK4 error ~ O(h^4) ~ 1e-14 relative.
    # This uses 4*10000 = 40000 calls.
    steps = 10000
    h = (t1 - t0) / steps
    t = t0
    y = y0_arr.copy()

    for _ in range(steps):
        if time.time() - start_time > 280:
            break

        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + h / 2.0, y + h / 2.0 * k1), dtype=float)
        k3 = np.asarray(f(t + h / 2.0, y + h / 2.0 * k2), dtype=float)
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float)

        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h

    return y
# EVOLVE-BLOCK-END
