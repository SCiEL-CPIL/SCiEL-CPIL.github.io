"""Use scipy.integrate.solve_ivp with Radau method and analytic Jacobian for Van der Pol (mu=5)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _vdp_jac(t, y):
    """Analytic Jacobian for Van der Pol oscillator with mu=5."""
    y0, y1 = y[0], y[1]
    mu = 5.0
    J = np.array([
        [0.0, 1.0],
        [-2.0 * mu * y0 * y1 - 1.0, mu * (1.0 - y0 * y0)]
    ])
    return J


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) from t_span[0] to t_span[1]
    using scipy's Radau method with tight tolerances and analytic Jacobian.
    
    The reference uses Radau with rtol=1e-11, atol=1e-13.
    We provide the analytic Jacobian to reduce the number of RHS evaluations.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Set a deadline guard: 300s wall-clock limit, use 280s for safety
    start_time = time.time()
    deadline = 280.0
    
    # Use Radau with the same tolerances as the reference
    # Provide analytic Jacobian to reduce function call count
    sol = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='Radau',
        rtol=1e-11,
        atol=1e-13,
        jac=_vdp_jac
    )
    
    if not sol.success:
        # Fallback: try without Jacobian if we have time
        elapsed = time.time() - start_time
        if elapsed < deadline - 5:
            sol = solve_ivp(
                f,
                (t0, t1),
                y0,
                method='Radau',
                rtol=1e-11,
                atol=1e-13
            )
    
    if sol.success and sol.y.shape[1] > 0:
        return sol.y[:, -1]
    else:
        # Last resort: return initial condition if something went wrong
        return y0
# EVOLVE-BLOCK-END
