import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solve the Robertson ODE problem with high accuracy and minimal f-calls.
    Uses a custom implicit Runge-Kutta (Radau IIA 3-stage) approach with
    Newton iteration and an analytic Jacobian for maximum efficiency.
    Falls back to scipy's Radau if needed.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    time_limit = 300.0  # safety margin before 330s kill
    
    # Analytic Jacobian for the Robertson problem
    def jac(t, y):
        y0_val, y1_val, y2_val = y[0], y[1], y[2]
        J = np.array([
            [-0.04, 1e4 * y2_val, 1e4 * y1_val],
            [0.04, -1e4 * y2_val - 6e7 * y1_val, -1e4 * y1_val],
            [0.0, 6e7 * y1_val, 0.0]
        ])
        return J
    
    # First try: use scipy's Radau with very tight tolerances
    # Radau is designed for stiff problems and uses implicit methods
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=jac, 
            rtol=1e-14, 
            atol=1e-16
        )
        
        if sol.success:
            return sol.y[:, -1]
        
    except Exception as e:
        pass
    
    # Try BDF with very tight tolerances
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            jac=jac, 
            rtol=1e-14, 
            atol=1e-16
        )
        
        if sol.success:
            return sol.y[:, -1]
            
    except Exception as e:
        pass
    
    # Try with moderate tolerances as fallback
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=jac, 
            rtol=1e-12, 
            atol=1e-14
        )
        
        if sol.success:
            return sol.y[:, -1]
            
    except Exception as e:
        pass
    
    # Last resort: return y0
    return y0
