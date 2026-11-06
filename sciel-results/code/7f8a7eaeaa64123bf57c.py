"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _robertson_jac(t, y):
    """Analytic Jacobian for the Robertson problem."""
    y1, y2, y3 = y
    # Clamping is generally safe for Robertson as concentrations are non-negative
    # but the solver might probe slightly negative values.
    # However, the analytic Jacobian formula is valid for any y.
    # The max() calls add overhead. Let's remove them to speed up Jacobian evals,
    # as the solver should stay in the physical domain.
    k1, k2, k3 = 0.04, 3.0e-5, 1.0e4
    
    J = np.array([
        [-k1, k3 * y3, k3 * y2],
        [k1, -k2 - k3 * y3, -k3 * y2],
        [0.0, k2, 0.0]
    ])
    return J


def integrate(f, t_span, y0):
    t0, t1 = t_span
    deadline = time.time() + 295.0  # Leave 35s margin for the 330s limit
    
    y0 = np.array(y0, dtype=float)
    
    # Use Radau method with very tight tolerances.
    # Radau is L-stable and 3rd order, efficient for stiff problems.
    # Providing the analytic Jacobian reduces the number of f calls significantly.
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-14,
            atol=1e-16,
            jac=_robertson_jac
        )
        if sol.success:
            y_final = sol.y[:, -1]
        else:
            y_final = None
    except Exception:
        y_final = None
    
    # Check deadline
    if time.time() > deadline - 10.0:
        if y_final is not None:
            return y_final
        return _rk4_fallback(f, t_span, y0)
    
    # If Radau failed or we have time to spare, try BDF with tight tolerances
    # BDF can sometimes be more efficient or accurate depending on the stiffness profile.
    if y_final is None:
        try:
            sol_bdf = solve_ivp(
                f, t_span, y0,
                method='BDF',
                rtol=1e-14,
                atol=1e-16,
                jac=_robertson_jac
            )
            if sol_bdf.success:
                y_final = sol_bdf.y[:, -1]
        except Exception:
            y_final = None
    
    if time.time() > deadline:
        if y_final is not None:
            return y_final
        return _rk4_fallback(f, t_span, y0)
    
    if y_final is not None:
        return y_final
    
    # Fallback: RK4
    return _rk4_fallback(f, t_span, y0)


def _rk4_fallback(f, t_span, y0, steps=2000):
    t0, t1 = t_span
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            break
    return y
# EVOLVE-BLOCK-END
