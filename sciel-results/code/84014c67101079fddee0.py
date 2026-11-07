"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) from t_span[0] to t_span[1]
    with initial condition y0.

    Uses Radau with the analytic Jacobian to minimize f-calls while
    matching the reference (Radau, rtol=1e-11, atol=1e-13) accuracy.
    """
    start_time = time.time()
    DEADLINE = 295.0

    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)

    call_count = [0]

    def counted_f(t, y):
        call_count[0] += 1
        return f(t, y)

    # Analytic Jacobian for Van der Pol: y' = [y1, mu*(1-y0^2)*y1 - y0]
    # J = [[0, 1],
    #      [-2*mu*y0*y1 - 1, mu*(1-y0^2)]]
    def jac(t, y):
        y0v, y1v = y[0], y[1]
        mu = 5.0
        return np.array([
            [0.0, 1.0],
            [-2.0 * mu * y0v * y1v - 1.0, mu * (1.0 - y0v * y0v)]
        ])

    def time_left():
        return (DEADLINE - (time.time() - start_time)) > 1.0

    def run(method, rtol, atol, use_jac=True):
        kwargs = {}
        if use_jac:
            kwargs['jac'] = jac
        sol = solve_ivp(
            counted_f,
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            dense_output=False,
            **kwargs
        )
        if sol.success:
            y = sol.y[:, -1]
            if np.all(np.isfinite(y)):
                return y
        return None

    # Primary: Radau with analytic Jacobian at reference tolerances
    y_best = run('Radau', 1e-11, 1e-13, use_jac=True)
    if y_best is None:
        y_best = run('Radau', 1e-11, 1e-13, use_jac=False)
    if y_best is None:
        y_best = run('Radau', 1e-10, 1e-12, use_jac=True)
    if y_best is None and time_left():
        y_best = run('RK45', 1e-9, 1e-11, use_jac=False)
    if y_best is None:
        return y0

    # Polish with tighter tolerances if time permits
    if time_left():
        y_polish = run('Radau', 1e-13, 1e-15, use_jac=True)
        if y_polish is not None and np.all(np.isfinite(y_polish)):
            y_best = y_polish

    return y_best
# EVOLVE-BLOCK-END
