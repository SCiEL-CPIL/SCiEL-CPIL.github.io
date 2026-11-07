"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the IVP from t_span[0] to t_span[1] with initial condition y0.
    Uses scipy's DOP853 with very tight tolerances to achieve high accuracy
    with a minimal number of function calls.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)

    # If the interval is trivial, just return y0
    if t0 == t1:
        return y0.copy()

    # Try using scipy's DOP853 with very tight tolerances
    try:
        from scipy.integrate import solve_ivp

        # DOP853 is an 8th order method (7th order accurate, 8th order error estimate)
        # Using extremely tight tolerances to maximize accuracy
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
            # Fallback to a fixed-step high-order RK if scipy fails
            return _rk4_fallback(f, t0, t1, y0, 5000)

        return sol.y[:, -1]

    except Exception:
        # Fallback to a fixed-step RK4 if scipy fails or is not available
        return _rk4_fallback(f, t0, t1, y0, 5000)


def _rk4_fallback(f, t0, t1, y0, steps):
    """Simple RK4 fallback with many steps."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
    return y

# EVOLVE-BLOCK-END
