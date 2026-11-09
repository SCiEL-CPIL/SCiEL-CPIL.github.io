"""ODE integrator for the Van der Pol oscillator using scipy Radau with analytic Jacobian."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _jacobian(t, y):
    """Analytic Jacobian of the Van der Pol system (mu=5)."""
    y0, y1 = y
    return np.array([
        [0.0, 1.0],
        [-2.0 * 5.0 * y0 * y1 - 1.0, 5.0 * (1.0 - y0 * y0)]
    ])


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using scipy Radau with analytic Jacobian.
    Returns the state at t_end.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use solve_ivp with Radau method, tight tolerances matching the reference
    # Provide analytic Jacobian to reduce f evaluations
    result = solve_ivp(
        f,
        t_span=(t0, t1),
        y0=y0,
        method='Radau',
        jac=_jacobian,
        rtol=1e-11,
        atol=1e-13,
        max_step=0.1
    )
    
    if not result.success:
        # Fallback: try with slightly looser tolerances if needed
        result = solve_ivp(
            f,
            t_span=(t0, t1),
            y0=y0,
            method='Radau',
            jac=_jacobian,
            rtol=1e-10,
            atol=1e-12,
            max_step=0.1
        )
    
    return result.y[:, -1]
# EVOLVE-BLOCK-END
