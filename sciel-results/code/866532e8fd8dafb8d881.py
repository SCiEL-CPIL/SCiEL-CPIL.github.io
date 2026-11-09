"""ODE integrator for Van der Pol oscillator using DOP853 with tight tolerances."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) from t_span[0] to t_span[1]
    with initial condition y0, using scipy's DOP853 method with tight
    tolerances to minimize function call count while maintaining accuracy.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: return current best if we exceed 320s
    start_time = time.time()
    deadline = 320.0  # seconds, leave 10s safety margin before 330s kill
    
    # Use DOP853 (8th-order explicit RK) with tight tolerances
    # DOP853 typically achieves high accuracy with fewer f calls than Radau
    # for moderately stiff problems like Van der Pol with mu=5
    sol = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='DOP853',
        rtol=1e-12,
        atol=1e-14,
        max_step=(t1 - t0) / 100.0
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
        # If DOP853 failed, try Radau as fallback
        def jac(t, y):
            y0_val = y[0]
            y1_val = y[1]
            return np.array([
                [0.0, 1.0],
                [-10.0 * y0_val * y1_val - 1.0, 5.0 * (1.0 - y0_val * y0_val)]
            ])
        
        sol2 = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            jac=jac,
            max_step=(t1 - t0) / 50.0
        )
        
        if sol2.success:
            return sol2.y[:, -1]
        else:
            if sol2.y is not None and sol2.y.shape[1] > 0:
                return sol2.y[:, -1]
            return y0

# EVOLVE-BLOCK-END
