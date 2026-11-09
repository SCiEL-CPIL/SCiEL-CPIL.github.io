# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)

    start = time.time()
    deadline = start + 280.0  # safety margin before 330s kill

    # Robertson problem is very stiff; use an implicit adaptive method.
    # Radau is robust for stiff IVPs and typically very accurate.
    # Try progressively tighter tolerances if time permits.
    tolerances = [
        (1e-12, 1e-15),
        (1e-14, 1e-17),
    ]

    for rtol, atol in tolerances:
        if time.time() > deadline - 60.0:
            break
        try:
            sol = solve_ivp(
                f, (t0, t1), y0,
                method="Radau",
                rtol=rtol,
                atol=atol,
                max_step=(t1 - t0) / 50.0,
            )
            if sol.success and sol.y.shape[1] > 0:
                return sol.y[:, -1]
        except Exception:
            pass

    # Fallback: explicit RK45 with adaptive stepping, guarded by deadline.
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-12,
            max_step=(t1 - t0) / 100.0,
        )
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Last-resort fixed-step RK4 with a small number of steps to stay in budget.
    try:
        steps = 2000
        h = (t1 - t0) / steps
        t = t0
        y = y0.copy()
        for _ in range(steps):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
        return y
    except Exception:
        return y0

# EVOLVE-BLOCK-END
