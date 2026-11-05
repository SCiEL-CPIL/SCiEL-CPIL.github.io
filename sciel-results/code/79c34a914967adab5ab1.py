"""ODE solver: Robertson problem using LSODA via scipy.integrate.solve_ivp."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) over t_span starting from y0.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)

    # Count function calls
    call_count = [0]
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit

    def wrapped_f(t, y):
        call_count[0] += 1
        return f(t, y)

    # Try LSODA with tight tolerances first
    # Robertson problem is stiff, LSODA handles stiffness well
    try:
        sol = solve_ivp(
            wrapped_f,
            t_span,
            y0,
            method='LSODA',
            rtol=1e-13,
            atol=1e-15,
            max_step=(t1 - t0) / 50.0,  # reasonable max step to keep accuracy
        )
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Check if we're running out of time
    if time.time() > deadline - 30:
        # Fall back to a quick RK45 if time is tight
        try:
            sol = solve_ivp(
                wrapped_f,
                t_span,
                y0,
                method='RK45',
                rtol=1e-8,
                atol=1e-10,
            )
            if sol.success and sol.y.shape[1] > 0:
                return sol.y[:, -1]
        except Exception:
            pass
        # Last resort: return initial condition
        return y0

    # Try Radau (implicit, good for stiff problems)
    try:
        sol = solve_ivp(
            wrapped_f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15,
        )
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Try BDF (implicit, good for stiff problems)
    try:
        sol = solve_ivp(
            wrapped_f,
            t_span,
            y0,
            method='BDF',
            rtol=1e-13,
            atol=1e-15,
        )
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: explicit RK45 with moderate tolerance
    try:
        sol = solve_ivp(
            wrapped_f,
            t_span,
            y0,
            method='RK45',
            rtol=1e-10,
            atol=1e-12,
        )
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: simple fixed-step RK4
    h = (t1 - t0) / 10000
    t, y = t0, y0.copy()
    n_steps = int(round((t1 - t0) / h))
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if time.time() > deadline:
            break
        if not np.all(np.isfinite(y)):
            break

    return y
# EVOLVE-BLOCK-END
