import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """Integrate y' = f(t, y) over t_span and return y at t_end."""
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    n = y0_arr.size
    deadline = time.time() + 280.0

    # Primary approach: high-order adaptive RK via solve_ivp (DOP853)
    # This is a 12-stage, 8th-order method with 7th-order error estimation.
    # For smooth problems like convection-diffusion, it achieves high accuracy
    # with relatively few function evaluations.
    # No max_step constraint: let the adaptive stepper choose optimal steps.

    best_y = None
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15,
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            best_y = sol.y[:, -1].copy()
    except Exception:
        pass

    # Polish: if we have time, run again with even tighter tolerances
    # using the previous result as initial guess won't help (same IVP),
    # but tighter tolerances with the same method can reduce error further.
    if best_y is not None and time.time() < deadline - 30.0:
        try:
            sol2 = solve_ivp(
                f,
                t_span,
                y0_arr,
                method='DOP853',
                rtol=1e-14,
                atol=1e-16,
            )
            if sol2.success and np.all(np.isfinite(sol2.y[:, -1])):
                best_y = sol2.y[:, -1].copy()
        except Exception:
            pass

    if best_y is not None:
        return best_y

    # Fallback: RK45 with tighter tolerances
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='RK45',
            rtol=1e-12,
            atol=1e-14,
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: fixed-step RK4 with many steps
    # Use as many steps as feasible within remaining time
    remaining = deadline - time.time()
    # Estimate: each RK4 step calls f 4 times, each call ~O(n) ops
    # Assume ~50us per f call for n=100 -> 200us per step
    # In remaining seconds, we can do remaining / 200e-6 steps
    max_steps = int(remaining * 5000)
    max_steps = max(max_steps, 1000)
    max_steps = min(max_steps, 2000000 // 4)  # stay under 2M f calls total

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
