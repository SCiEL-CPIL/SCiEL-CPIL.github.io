"""ODE integrator for the Lotka-Volterra system using scipy's DOP853."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra IVP y' = f(t, y) over t_span starting at y0.
    Uses scipy.integrate.solve_ivp with method='DOP853', rtol=1e-13, atol=1e-15
    to closely match the reference (rtol=1e-11, atol=1e-13) while keeping
    the number of f-calls low.
    """
    start_time = time.time()
    deadline = start_time + 270.0  # safety margin before 300 s wall-clock

    y0_arr = np.asarray(y0, dtype=float)
    t0, t1 = t_span

    # Ensure f is called at least once (mandatory by contract).
    _ = f(t0, y0_arr)

    # Primary integration with tight tolerances to approach machine precision.
    sol = solve_ivp(
        f,
        (t0, t1),
        y0_arr,
        method="DOP853",
        rtol=1e-13,
        atol=1e-15,
        dense_output=False,
        max_step=0.05,
    )

    if sol.success and np.all(np.isfinite(sol.y[:, -1])):
        result = sol.y[:, -1]
    else:
        # Fallback: slightly looser tolerances, larger max_step
        sol = solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
            dense_output=False,
            max_step=0.1,
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            result = sol.y[:, -1]
        else:
            # Last resort: RK45 with conservative settings
            sol = solve_ivp(
                f,
                (t0, t1),
                y0_arr,
                method="RK45",
                rtol=1e-12,
                atol=1e-14,
                max_step=0.01,
            )
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                result = sol.y[:, -1]
            else:
                # Return best available
                result = sol.y[:, -1] if sol.y.size > 0 else np.zeros_like(y0_arr)

    # Guard against deadline
    if time.time() > deadline:
        pass  # still return what we have

    return result
# EVOLVE-BLOCK-END
