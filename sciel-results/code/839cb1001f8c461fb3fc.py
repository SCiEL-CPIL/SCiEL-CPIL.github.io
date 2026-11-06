"""High-accuracy Robertson ODE solver using LSODA with analytic Jacobian.

Strategy:
  - Use scipy.integrate.odeint (which wraps LSODA) or scipy.integrate.solve_ivp
    with LSODA method for the stiff Robertson problem.
  - Provide the analytic Jacobian to reduce function evaluations.
  - Use very tight tolerances to minimize relative error.
  - Keep a deadline guard to avoid exceeding the 300s limit.

The Robertson problem is highly stiff, so implicit stiff solvers are essential.
LSODA automatically switches between Adams (non-stiff) and BDF (stiff) methods.
"""

import numpy as np
import time
from scipy.integrate import solve_ivp


def _robertson_jacobian(t, y):
    """Analytic Jacobian for the Robertson ODE."""
    y = np.asarray(y, dtype=float).reshape(-1)
    n = y.shape[0]
    J = np.zeros((n, n))
    if n >= 3:
        y0v, y1v, y2v = y[0], y[1], y[2]
        J[0, 0] = -0.04
        J[0, 1] = 1.0e4 * y2v
        J[0, 2] = 1.0e4 * y1v
        J[1, 0] = 0.04
        J[1, 1] = -1.0e4 * y2v - 6.0e7 * y1v
        J[1, 2] = -1.0e4 * y1v
        J[2, 0] = 0.0
        J[2, 1] = 6.0e7 * y1v
        J[2, 2] = 0.0
    return J


def integrate(f, t_span, y0):
    """
    Solve the fixed Robertson IVP on [t0, t1] and return y(t1).

    Parameters
    ----------
    f : callable
        ODE right-hand side f(t, y) -> dy/dt.
    t_span : tuple
        (t0, t1) integration interval.
    y0 : array-like
        Initial condition at t0.

    Returns
    -------
    y_end : ndarray
        Final state y(t1).
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float).reshape(-1)
    n = y0_arr.shape[0]

    start_time = time.time()
    time_limit = 290.0  # safety margin before 300s kill

    def elapsed():
        return time.time() - start_time

    def time_left():
        return time_limit - elapsed()

    best_result = None

    # Try LSODA with very tight tolerances
    # LSODA is good for stiff problems and can be very accurate
    try:
        if time_left() > 10.0:
            sol = solve_ivp(
                f, (t0, t1), y0_arr,
                method='LSODA',
                jac=_robertson_jacobian,
                rtol=1e-12,
                atol=1e-14,
            )
            if sol.success and sol.y.shape[1] >= 1:
                best_result = sol.y[:, -1].copy()
    except Exception:
        pass

    # If LSODA didn't work, try Radau
    if best_result is None:
        try:
            if time_left() > 10.0:
                sol = solve_ivp(
                    f, (t0, t1), y0_arr,
                    method='Radau',
                    jac=_robertson_jacobian,
                    rtol=1e-12,
                    atol=1e-14,
                )
                if sol.success and sol.y.shape[1] >= 1:
                    best_result = sol.y[:, -1].copy()
        except Exception:
            pass

    # If Radau didn't work, try BDF
    if best_result is None:
        try:
            if time_left() > 10.0:
                sol = solve_ivp(
                    f, (t0, t1), y0_arr,
                    method='BDF',
                    jac=_robertson_jacobian,
                    rtol=1e-12,
                    atol=1e-14,
                )
                if sol.success and sol.y.shape[1] >= 1:
                    best_result = sol.y[:, -1].copy()
        except Exception:
            pass

    # Last resort: return y0
    if best_result is None:
        best_result = y0_arr.copy()

    return best_result
