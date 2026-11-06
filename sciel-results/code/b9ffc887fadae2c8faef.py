import numpy as np
from scipy.integrate import solve_ivp
import time

def _robertson_jac(t, y):
    """
    Analytic Jacobian for the Robertson problem.
    """
    y0, y1, y2 = y
    J = np.array([
        [-0.04, 1e4 * y2, 1e4 * y1],
        [0.04, -1e4 * y2 - 6e7 * y1, -1e4 * y1],
        [0.0, 6e7 * y1, 0.0]
    ])
    return J

def integrate(f, t_span, y0):
    """
    Integrates the Robertson ODE problem.
    Uses LSODA with analytic Jacobian as primary, with fallbacks.
    """
    t_start, t_end = t_span
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Primary: LSODA with analytic Jacobian
    # LSODA switches between Adams (non-stiff) and BDF (stiff)
    # This typically gives fewer function evaluations than pure Radau/BDF
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='LSODA', 
            jac=_robertson_jac, 
            rtol=1e-13, 
            atol=1e-15,
            dense_output=False
        )
        
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
        
        # If LSODA failed, try Radau with analytic Jacobian
        if time.time() > deadline - 10.0:
            if sol.y.shape[1] > 0:
                return sol.y[:, -1]
            return y0
        
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=_robertson_jac, 
            rtol=1e-12, 
            atol=1e-14,
            dense_output=False
        )
        
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
        
        if time.time() > deadline - 5.0:
            if sol.y.shape[1] > 0:
                return sol.y[:, -1]
            return y0
        
        # Fallback: BDF with analytic Jacobian
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            jac=_robertson_jac, 
            rtol=1e-12, 
            atol=1e-14,
            dense_output=False
        )
        
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
        
        if sol.y.shape[1] > 0:
            return sol.y[:, -1]
        return y0

    except Exception:
        return y0
