"""ODE solver: Robertson problem using Radau with analytical Jacobian."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    
    Uses Radau (implicit Runge-Kutta) with an analytical Jacobian for the
    Robertson problem to minimize function evaluations while achieving
    high accuracy.
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Deadline guard: 300s to leave margin for the evaluator
    deadline = time.time() + 300.0
    
    # Count function calls to monitor budget (max 2,000,000)
    call_count = [0]
    
    def wrapped_f(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Analytical Jacobian for the Robertson problem
    # y' = f(t, y) where:
    # y[0]' = -0.04*y[0] + 1e4*y[1]*y[2]
    # y[1]' = 0.04*y[0] - 1e4*y[1]*y[2] - 3e7*y[1]^2
    # y[2]' = 3e7*y[1]^2
    def jac(t, y):
        # Jacobian is independent of t for Robertson
        J = np.array([
            [-0.04, 1e4*y[2], 1e4*y[1]],
            [0.04, -1e4*y[2] - 6e7*y[1], -1e4*y[1]],
            [0.0, 6e7*y[1], 0.0]
        ])
        return J
    
    # Primary solve: Radau with tight tolerances and analytical Jacobian
    # The Jacobian significantly reduces the number of f evaluations needed
    # for the implicit solver to converge
    sol = solve_ivp(
        wrapped_f, t_span, y0_arr,
        method='Radau',
        jac=jac,
        rtol=1e-13,
        atol=1e-15
    )
    
    if not sol.success:
        # Fallback: try with coarser settings
        sol = solve_ivp(
            wrapped_f, t_span, y0_arr,
            method='Radau',
            jac=jac,
            rtol=1e-11,
            atol=1e-13
        )
        if not sol.success:
            return sol.y[:, -1]
    
    y_current = sol.y[:, -1]
    
    # Check if we have time for a refinement pass
    if time.time() < deadline - 30.0 and call_count[0] < 1500000:
        # Refine with even tighter tolerance
        sol2 = solve_ivp(
            wrapped_f, t_span, y0_arr,
            method='Radau',
            jac=jac,
            rtol=1e-14,
            atol=1e-16
        )
        if sol2.success:
            y_current = sol2.y[:, -1]
    
    return y_current
# EVOLVE-BLOCK-END
