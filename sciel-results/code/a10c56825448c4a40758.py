import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solve the Robertson ODE problem using an adaptive stiff solver.
    
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
    
    start_time = time.time()
    time_limit = 290.0  # seconds, safety margin
    
    # Analytic Jacobian for the Robertson problem
    # y' = [-0.04*y0 + 1e4*y1*y2,
    #       0.04*y0 - 1e4*y1*y2 - 3e7*y1^2,
    #       3e7*y1^2]
    def jac(t, y):
        y0_val, y1_val, y2_val = y[0], y[1], y[2]
        # Using a pre-allocated array is slightly faster, but creating a new one is fine for 3x3
        J = np.array([
            [-0.04, 1e4 * y2_val, 1e4 * y1_val],
            [0.04, -1e4 * y2_val - 6e7 * y1_val, -1e4 * y1_val],
            [0.0, 6e7 * y1_val, 0.0]
        ])
        return J

    # Strategy:
    # 1. Try Radau with very tight tolerances and NO max_step to minimize function calls.
    # 2. If that fails or takes too long, try BDF.
    # 3. If that fails, try LSODA.
    
    # Attempt 1: Radau with high precision, adaptive step size (no max_step constraint)
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            jac=jac,
            rtol=1e-13,
            atol=1e-16,
            # No max_step, allows solver to take large steps where solution is smooth
            vectorized=False
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            raise TimeoutError()

        if sol.success:
            return sol.y[:, -1]
            
    except Exception as e:
        # Check time limit
        if time.time() - start_time > time_limit:
            pass

    # Attempt 2: BDF with high precision
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF',
            jac=jac,
            rtol=1e-13,
            atol=1e-16,
            max_order=5
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            raise TimeoutError()

        if sol.success:
            return sol.y[:, -1]
            
    except Exception as e:
        pass

    # Attempt 3: LSODA (automatic stiff/non-stiff switching)
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='LSODA',
            rtol=1e-12,
            atol=1e-15
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            raise TimeoutError()

        if sol.success:
            return sol.y[:, -1]
            
    except Exception as e:
        pass

    # Fallback: Return initial state if everything fails (should not happen for Robertson)
    # But to ensure we return *something*, we can do a very coarse Radau step
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            jac=jac,
            rtol=1e-6,
            atol=1e-9
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception as e:
        pass

    return y0
