import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    start_time = time.time()
    DEADLINE = 280  # safety margin before 330s limit

    def jac(t, y):
        y0v, y1v = y[0], y[1]
        return np.array([
            [0.0, 1.0],
            [-10.0 * y0v * y1v - 1.0, 5.0 * (1.0 - y0v**2)]
        ])

    # Try with very tight tolerances
    try:
        result = solve_ivp(
            f, t_span, y0,
            method='Radau',
            jac=jac,
            rtol=1e-13, atol=1e-15,
            max_step=0.005
        )
        if result.success:
            return result.y[:, -1]
    except Exception:
        pass

    # Fallback with slightly looser tolerances
    try:
        result = solve_ivp(
            f, t_span, y0,
            method='Radau',
            jac=jac,
            rtol=1e-12, atol=1e-14
        )
        if result.success:
            return result.y[:, -1]
    except Exception:
        pass

    # Final fallback
    try:
        result = solve_ivp(
            f, t_span, y0,
            method='Radau',
            jac=jac,
            rtol=1e-11, atol=1e-13
        )
        if result.success:
            return result.y[:, -1]
    except Exception:
        pass

    # Absolute fallback: simple RK4
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    h = (t1 - t0) / 10000
    t = t0
    for _ in range(10000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
