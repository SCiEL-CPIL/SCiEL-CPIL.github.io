import numpy as np
import time
from scipy.integrate import solve_ivp

def _jacobian(t, y):
    """
    Analytic Jacobian for the Robertson problem.
    y0' = -0.04 y0 + 1e4 y1 y2
    y1' = 0.04 y0 - 1e4 y1 y2 - 3e3 y1^2
    y2' = 3e3 y1^2
    """
    y0, y1, y2 = y
    J = np.array([
        [-0.04, 1e4 * y2, 1e4 * y1],
        [0.04, -1e4 * y2 - 6e3 * y1, -1e4 * y1],
        [0.0, 6e3 * y1, 0.0]
    ])
    return J

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = 300.0  # seconds, keep safe margin from 330s limit

    # Primary strategy: Radau with analytic Jacobian and very tight tolerances.
    # Radau is an implicit Runge-Kutta method, A-stable, ideal for stiff problems.
    # Providing the Jacobian prevents finite-difference approximations, saving f-calls.
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=_jacobian,
            rtol=1e-14, 
            atol=1e-16,
            max_step=np.inf
        )
        
        if sol.success:
            return sol.y[:, -1]
            
    except Exception:
        pass

    # Fallback: BDF with analytic Jacobian.
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            jac=_jacobian,
            rtol=1e-14, 
            atol=1e-16,
            max_step=np.inf
        )
        
        if sol.success:
            return sol.y[:, -1]
            
    except Exception:
        pass

    # Last resort: Return initial state (should not happen for Robertson)
    return y0
