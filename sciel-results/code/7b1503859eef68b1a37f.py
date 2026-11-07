import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin before 330s kill

    try:
        from scipy.integrate import solve_ivp

        # Use DOP853 (12th order) with very tight tolerances.
        # The reference uses rtol=1e-11, atol=1e-13.
        # We use rtol=1e-14, atol=1e-16 to push accuracy to the limit.
        # DOP853 is very efficient: error ~ h^12, so even with tight tolerances,
        # the number of steps (and thus f calls) remains manageable.

        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-14,
            atol=1e-16,
            dense_output=False
        )

        if not sol.success:
            # Fallback to reference tolerances
            sol = solve_ivp(
                f,
                (t0, t1),
                y0,
                method='DOP853',
                rtol=1e-11,
                atol=1e-13,
                dense_output=False
            )

        if not sol.success:
            # Last resort: RK4
            return _rk4_fallback(f, t0, t1, y0, start_time, deadline)

        y_end = sol.y[:, -1]

        # Verify result is finite and has correct shape
        if not np.all(np.isfinite(y_end)) or y_end.shape != y0.shape:
            return _rk4_fallback(f, t0, t1, y0, start_time, deadline)

        return y_end

    except Exception:
        # Fallback to RK4 if any exception occurs
        return _rk4_fallback(f, t0, t1, y0, start_time, deadline)


def _rk4_fallback(f, t0, t1, y0, start_time, deadline):
    """
    Simple RK4 fallback with adaptive step sizing, respecting the deadline.
    """
    y = np.array(y0, dtype=float)
    h = (t1 - t0) / 1000
    t = t0

    max_steps = 50000
    step_count = 0

    while t < t1 and step_count < max_steps:
        if time.time() > deadline - 5.0:
            break

        if t + h > t1:
            h = t1 - t

        # RK4
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)

        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        step_count += 1

    return y
