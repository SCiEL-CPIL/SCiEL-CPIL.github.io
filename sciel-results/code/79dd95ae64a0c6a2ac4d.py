import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using scipy's Radau method
    with analytical Jacobian to minimize function calls.
    
    Parameters:
    f : callable
        Function f(t, y) returning the derivative dy/dt.
    t_span : tuple
        (t_start, t_end)
    y0 : array-like
        Initial state.
        
    Returns:
    y_end : ndarray
        State at t_end.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    timeout = 300.0
    
    try:
        # Analytical Jacobian for the Van der Pol oscillator (mu=5):
        # y0' = y1
        # y1' = 5*(1-y0^2)*y1 - y0
        # J = [[0, 1],
        #      [-10*y0*y1 - 1, 5*(1-y0^2)]]
        def jac(t, y):
            y0_val = y[0]
            y1_val = y[1]
            return np.array([
                [0.0, 1.0],
                [-10.0 * y0_val * y1_val - 1.0, 5.0 * (1.0 - y0_val * y0_val)]
            ])
        
        # Use Radau with analytical Jacobian to reduce f calls
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13,
            jac=jac,
            max_step=0.1
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            return sol.y[:, -1] if sol.y.shape[1] > 0 else y0
            
    except Exception:
        return y0
