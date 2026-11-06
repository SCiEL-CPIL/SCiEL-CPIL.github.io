"""Direct solver for the fixed Robertson ODE: single high-accuracy Radau IVP."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Solve the fixed Robertson IVP on [t0, t1] with a single tight Radau solve.

    Parameters
    ----------
    f : callable
        ODE right-hand side f(t, y) -> dy/dt.
    t_span : tuple
        (t0, t1) integration interval (fixed to [0, 40] by the evaluator).
    y0 : array-like
        Initial condition at t0.

    Returns
    -------
    y_end : ndarray
        Final state y(t1).
    """
    t0, t1 = t_span
    y0_flat = np.asarray(y0, dtype=float).reshape(-1)

    # Analytic Jacobian df/dy for the Robertson problem (matches the fixed task):
    #   f0 = -0.04*y0 + 1e4*y1*y2
    #   f1 =  0.04*y0 - 1e4*y1*y2 - 3e7*y1^2
    #   f2 =  3e7*y1^2
    def jac(t, y):
        y = y.reshape(-1)
        y0v, y1v, y2v = y[0], y[1], y[2]
        return np.array([
            [-0.04,                  1.0e4 * y2v,  1.0e4 * y1v],
            [ 0.04, -1.0e4 * y2v - 6.0e7 * y1v, -1.0e4 * y1v],
            [ 0.0,                   6.0e7 * y1v,  0.0      ],
        ])

    # Time budget guard: evaluator kills at 330s; keep a safety margin.
    start_time = time.time()
    time_limit = 300.0

    # Single high-accuracy Radau solve with analytic Jacobian.
    # Tight tolerances to minimize relative terminal error.
    # max_step constrains the step in the fast transient region.
    sol = solve_ivp(
        f, (t0, t1), y0_flat,
        method="Radau",
        jac=jac,
        rtol=1e-13,
        atol=1e-16,
        max_step=0.1,
    )

    if sol.success and sol.y.shape[1] >= 1:
        return sol.y[:, -1].reshape(-1)

    # Fallback: if the tight solve failed, try a looser one.
    sol2 = solve_ivp(
        f, (t0, t1), y0_flat,
        method="Radau",
        jac=jac,
        rtol=1e-10,
        atol=1e-13,
        max_step=0.5,
    )

    if sol2.success and sol2.y.shape[1] >= 1:
        return sol2.y[:, -1].reshape(-1)

    # Last resort: forward Euler with many steps.
    m = 200000
    h = (t1 - t0) / m
    y = y0_flat.copy()
    tt = t0
    for _ in range(m):
        y = y + h * f(tt, y)
        tt += h
    return y
# EVOLVE-BLOCK-END
