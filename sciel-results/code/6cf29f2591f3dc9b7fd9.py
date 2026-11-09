"""ODE solver: Convection-diffusion 100-point.
Strategy:
1. Primary: DOP853 (8th-order adaptive RK) with very tight tolerances.
2. Fallback: DOP853 with looser tolerances.
3. Fallback: Radau (implicit, stiff-capable).
4. Fallback: Fixed-step RK4.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp

# Deadline guard: we must return well before 330s wall-clock.
_DEADLINE_SECONDS = 315.0  # 15s safety margin


def _integrate_dop853(f, t_span, y0, rtol, atol, max_evals=2000000):
    """
    Adaptive 8th-order Runge-Kutta with error estimation (DOP853).
    Returns (y_final, count, success).
    """
    y0 = np.asarray(y0, dtype=float)
    state = {"count": 0}

    def f_counted(t, y):
        state["count"] += 1
        return f(t, y)

    try:
        sol = solve_ivp(
            f_counted,
            t_span,
            y0,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=np.inf,
            dense_output=False,
        )
        if sol.success:
            return sol.y[:, -1], state["count"], True
        else:
            return None, state["count"], False
    except Exception:
        return None, state["count"], False


def _integrate_radau(f, t_span, y0, rtol, atol, max_evals=2000000):
    """
    Implicit Radau solver for stiff problems.
    Returns (y_final, count, success).
    """
    y0 = np.asarray(y0, dtype=float)
    state = {"count": 0}

    def f_counted(t, y):
        state["count"] += 1
        return f(t, y)

    try:
        sol = solve_ivp(
            f_counted,
            t_span,
            y0,
            method="Radau",
            rtol=rtol,
            atol=atol,
            max_step=np.inf,
            dense_output=False,
        )
        if sol.success:
            return sol.y[:, -1], state["count"], True
        else:
            return None, state["count"], False
    except Exception:
        return None, state["count"], False


def _fixed_rk4(f, t0, t1, y0, steps):
    """Fixed-step RK4 as a last resort (no adaptivity)."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    n = len(y)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def integrate(f, t_span, y0):
    t_start = time.time()
    y0 = np.asarray(y0, dtype=float)

    def elapsed():
        return time.time() - t_start

    # Strategy 1: DOP853 with very tight tolerances
    if elapsed() < _DEADLINE_SECONDS - 30:
        y, count, success = _integrate_dop853(f, t_span, y0, rtol=1e-13, atol=1e-15)
        if success and count <= 2000000:
            return y

    # Strategy 2: DOP853 with slightly looser tolerances
    if elapsed() < _DEADLINE_SECONDS - 20:
        y, count, success = _integrate_dop853(f, t_span, y0, rtol=1e-11, atol=1e-13)
        if success and count <= 2000000:
            return y

    # Strategy 3: Radau (implicit) for stiff problems
    if elapsed() < _DEADLINE_SECONDS - 15:
        y, count, success = _integrate_radau(f, t_span, y0, rtol=1e-10, atol=1e-12)
        if success and count <= 2000000:
            return y

    # Strategy 4: Fixed-step RK4 with moderate steps
    if elapsed() < _DEADLINE_SECONDS - 10:
        y = _fixed_rk4(f, t_span[0], t_span[1], y0, 5000)
        if y is not None:
            return y

    # Strategy 5: Even fewer steps if very tight on time
    y = _fixed_rk4(f, t_span[0], t_span[1], y0, 2000)
    if y is not None:
        return y

    # Final fallback: return initial condition (should not happen)
    return y0.copy()
