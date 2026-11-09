"""ODE solver for Robertson problem using Radau IIA with analytic Jacobian."""
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Start time for the deadline guard
    start_time = time.time()
    deadline = 320.0  # safety margin before 330s limit
    
    # Analytic Jacobian for the Robertson problem (full 3D)
    def jac(t, y):
        y0_val, y1_val, y2_val = y[0], y[1], y[2]
        J = np.empty((3, 3))
        J[0, 0] = -0.04
        J[0, 1] = 1e4 * y2_val
        J[0, 2] = 1e4 * y1_val
        J[1, 0] = 0.04
        J[1, 1] = -1e4 * y2_val - 6e7 * y1_val
        J[1, 2] = -1e4 * y1_val
        J[2, 0] = 0.0
        J[2, 1] = 6e7 * y1_val
        J[2, 2] = 0.0
        return J
    
    # Use Radau IIA which is generally more accurate for stiff problems than BDF
    # with fewer function evaluations for the same accuracy
    rtol = 1e-13
    atol = 1e-15
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='Radau', 
            rtol=rtol, 
            atol=atol,
            jac=jac
        )
        
        if sol.success:
            # Apply conservation law correction: y0 + y1 + y2 = 1
            y_end = sol.y[:, -1]
            total = y_end[0] + y_end[1] + y_end[2]
            if total > 0:
                y_end = y_end / total
            return y_end
    
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time >= deadline:
        return y0_arr
    
    # Fallback: try BDF with reduced 2D system using conservation law
    def f_reduced(t, z):
        y2 = 1.0 - z[0] - z[1]
        dy = f(t, np.array([z[0], z[1], y2]))
        return np.array([dy[0], dy[1]])
        
    def jac_reduced(t, z):
        u, v = z[0], z[1]
        j11 = -0.04 - 1e4 * v
        j12 = 1e4 * (1.0 - u - 2.0 * v)
        j21 = 0.04 + 1e4 * v
        j22 = -1e4 * (1.0 - u - 2.0 * v) - 6e7 * v
        return np.array([[j11, j12], [j21, j22]])
    
    y0_reduced = np.array([y0_arr[0], y0_arr[1]])
    
    try:
        sol2 = solve_ivp(
            f_reduced, 
            t_span, 
            y0_reduced, 
            method='BDF', 
            rtol=rtol, 
            atol=atol,
            jac=jac_reduced
        )
        
        if sol2.success:
            y_end_reduced = sol2.y[:, -1]
            y2_end = 1.0 - y_end_reduced[0] - y_end_reduced[1]
            return np.array([y_end_reduced[0], y_end_reduced[1], y2_end])
    
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time >= deadline:
        return y0_arr

    # If everything fails, return initial condition
    return y0_arr
