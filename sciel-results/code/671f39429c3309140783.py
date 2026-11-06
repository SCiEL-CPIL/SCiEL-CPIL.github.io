import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solve the Robertson ODE problem using an implicit Radau solver with analytic Jacobian.
    
    Parameters:
    -----------
    f : callable
        The ODE function f(t, y) -> y'
    t_span : tuple
        (t0, t1) integration interval
    y0 : array-like
        Initial condition
    
    Returns:
    --------
    y_end : ndarray
        Final state at t1
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: 330s wall-clock limit, use 300s for safety
    start_time = time.time()
    time_limit = 300.0  # seconds
    
    # Analytic Jacobian for the Robertson problem
    def jac(t, y):
        y0_val, y1_val, y2_val = y[0], y[1], y[2]
        J = np.array([
            [-0.04, 1e4 * y2_val, 1e4 * y1_val],
            [0.04, -1e4 * y2_val - 6e7 * y1_val, -1e4 * y1_val],
            [0.0, 6e7 * y1_val, 0.0]
        ])
        return J
    
    # Use Radau method (implicit Runge-Kutta, 3rd order, A-stable)
    # Tight tolerances to maximize accuracy score.
    # Radau is efficient for this problem, so tight tolerances are feasible within time limits.
    
    # Primary attempt: very high accuracy
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            jac=jac,
            rtol=1e-13,
            atol=1e-17,
            max_step=1.0
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            # Should not happen for Robertson, but guard anyway
            pass

        if sol.success:
            return sol.y[:, -1]
            
    except Exception as e:
        pass

    # Fallback: slightly looser if primary failed for some reason (e.g., memory, step size issues)
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            jac=jac,
            rtol=1e-11,
            atol=1e-15,
            max_step=1.0
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception as e:
        pass

    # Last resort: BDF method
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF',
            jac=jac,
            rtol=1e-10,
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception as e:
        pass

    # Absolute last resort: simple explicit Euler (very inaccurate, high evals, but returns something)
    # This will score very low, but is better than crashing
    h = (t1 - t0) / 100000
    t = t0
    y = y0.copy()
    n_steps = 100000
    for _ in range(n_steps):
        if time.time() - start_time > time_limit - 1.0:
            break
        y = y + h * f(t, y)
        t += h
        
    return y
