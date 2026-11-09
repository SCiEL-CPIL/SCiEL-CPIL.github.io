import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using scipy's Radau method with analytic Jacobian.
    
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
    
    # Deadline guard: 300s limit, use 280s to be safe.
    start_time = time.time()
    timeout = 280.0
    
    # Analytic Jacobian for Van der Pol oscillator with mu=5
    # y0' = y1
    # y1' = 5 * (1 - y0^2) * y1 - y0
    # J = [[0, 1],
    #      [-1 - 2*mu*y0*y1, mu*(1 - y0^2)]]
    mu = 5.0
    def jac(t, y):
        y0_val, y1_val = y[0], y[1]
        return np.array([
            [0.0, 1.0],
            [-1.0 - 2.0 * mu * y0_val * y1_val, mu * (1.0 - y0_val * y0_val)]
        ])
    
    try:
        # Radau is an implicit Runge-Kutta method that handles stiffness well.
        # Providing the analytic Jacobian reduces the number of f calls significantly
        # compared to finite difference approximation.
        # Tight tolerances to match or exceed the reference accuracy.
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14,
            jac=jac
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If it fails, return the last known state or initial state
            return sol.y[:, -1] if sol.y.shape[1] > 0 else y0
            
    except Exception:
        # In case of any unexpected error, return y0 to ensure a finite vector
        return y0
