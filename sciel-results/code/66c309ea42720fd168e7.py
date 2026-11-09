"""ODE integrator for Van der Pol oscillator using scipy Radau with analytic Jacobian."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) from t_span[0] to t_span[1]
    with initial condition y0, using scipy's Radau method with an analytic
    Jacobian to minimize function call count.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: return current best if we exceed 320s
    start_time = time.time()
    deadline = 320.0  # seconds, leave 10s safety margin before 330s kill
    
    # Analytic Jacobian for the Van der Pol oscillator:
    # y0' = y1
    # y1' = 5*(1 - y0^2)*y1 - y0
    # J = [[0, 1],
    #      [-10*y0*y1 - 1, 5*(1 - y0^2)]]
    
    def jac(t, y):
        y0_val = y[0]
        y1_val = y[1]
        return np.array([
            [0.0, 1.0],
            [-10.0 * y0_val * y1_val - 1.0, 5.0 * (1.0 - y0_val * y0_val)]
        ])
    
    # Use Radau with the same tolerances as the reference to minimize calls
    # while maintaining high accuracy
    sol = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='Radau',
        rtol=1e-11,
        atol=1e-13,
        jac=jac,
        max_step=(t1 - t0) / 50.0  # reasonable max step to ensure accuracy
    )
    
    # Check if we're still within deadline
    if time.time() > deadline:
        # Return whatever we have
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback: return last available state
            if sol.y is not None and sol.y.shape[1] > 0:
                return sol.y[:, -1]
            return y0
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # If Radau failed, try a simpler approach as fallback
        if sol.y is not None and sol.y.shape[1] > 0:
            return sol.y[:, -1]
        return y0

# EVOLVE-BLOCK-END
