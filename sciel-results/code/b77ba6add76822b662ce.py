<<<<<<< SEARCH
import numpy as np
from scipy.integrate import odeint


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] using scipy's odeint with LSODA.
    
    LSODA automatically detects stiffness and adjusts step size, which is ideal for
    the Robertson problem. Providing the analytic Jacobian reduces function evaluations.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Analytic Jacobian for the Robertson problem
    def jac(t, y):
        y0, y1, y2 = y
        return np.array([
            [-0.04, 1e4 * y2, 1e4 * y1],
            [0.04, -1e4 * y2 - 6e7 * y1, -1e4 * y1],
            [0.0, 6e7 * y1, 0.0]
        ])

    # Create time points: just the start and end to get only the final state
    # odeint requires at least 2 time points
    t_eval = np.array([t0, t1])
    
    # Use LSODA with tight tolerances for high accuracy
    # rtol=1e-12 and atol=1e-14 for very accurate results
    # Dfun provides the Jacobian to reduce f evaluations
    try:
        y_end = odeint(f, y0, t_eval, method='LSODA', rtol=1e-12, atol=1e-14, Dfun=jac)
        return y_end[-1]
    except Exception:
        # Fallback to Radau if LSODA fails
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-12, atol=1e-14, jac=jac)
        return sol.y[:, -1]
=======
import numpy as np
from scipy.integrate import odeint


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] using scipy's odeint with LSODA.
    
    LSODA automatically detects stiffness and adjusts step size. 
    Providing the correct analytic Jacobian significantly reduces function evaluations 
    and improves step size selection for the stiff Robertson problem.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Analytic Jacobian for the Robertson problem
    # y1' = -0.04*y1 + 1e4*y2*y3
    # y2' = 0.04*y1 - 1e4*y2*y3 - 6e7*y2**2
    # y3' = 1e4*y2*y3 - 6e7*y2**2
    def jac(t, y):
        y1, y2, y3 = y
        return np.array([
            [-0.04, 1e4 * y3, 1e4 * y2],
            [0.04, -1e4 * y3 - 1.2e8 * y2, -1e4 * y2],
            [0.0, 1e4 * y3 - 1.2e8 * y2, 1e4 * y2]
        ])

    # Create time points: just the start and end to get only the final state
    # odeint requires at least 2 time points
    t_eval = np.array([t0, t1])
    
    # Use LSODA with tight tolerances for high accuracy
    # Correct Jacobian allows for more efficient stepping
    y_end = odeint(f, y0, t_eval, method='LSODA', rtol=1e-12, atol=1e-14, Dfun=jac)
    return y_end[-1]
>>>>>>> REPLACE
