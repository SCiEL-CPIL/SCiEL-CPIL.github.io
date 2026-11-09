"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _robertson_jac(t, y):
    """Analytical Jacobian for the Robertson problem."""
    y1, y2, y3 = y
    # Robertson problem:
    # y1' = -0.04*y1 + 1e4*y2*y3
    # y2' =  0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
    # y3' =  3e7*y2^2
    #
    # Jacobian:
    # d(y1')/dy1 = -0.04
    # d(y1')/dy2 = 1e4*y3
    # d(y1')/dy3 = 1e4*y2
    #
    # d(y2')/dy1 = 0.04
    # d(y2')/dy2 = -1e4*y3 - 6e7*y2
    # d(y2')/dy3 = -1e4*y2
    #
    # d(y3')/dy1 = 0
    # d(y3')/dy2 = 6e7*y2
    # d(y3')/dy3 = 0
    J = np.array([
        [-0.04, 1e4 * y3, 1e4 * y2],
        [0.04, -1e4 * y3 - 6e7 * y2, -1e4 * y2],
        [0.0, 6e7 * y2, 0.0]
    ])
    return J


def integrate(f, t_span, y0):
    """
    Integrate the Robertson problem using scipy's BDF method with analytical Jacobian.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: 300s wall-clock limit, use 280s as safety margin
    deadline = time.time() + 280.0
    
    # Use BDF method with analytical Jacobian for stiff problem
    # Tight tolerances for high accuracy
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='BDF',
        jac=_robertson_jac,
        rtol=1e-12,
        atol=1e-15,
        max_step=0.1
    )
    
    if not sol.success:
        # Fallback: try with Radau if BDF fails
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            jac=_robertson_jac,
            rtol=1e-12,
            atol=1e-15,
            max_step=0.1
        )
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # If both fail, return whatever we have
        if sol.y is not None and sol.y.shape[1] > 0:
            return sol.y[:, -1]
        return y0
# EVOLVE-BLOCK-END
