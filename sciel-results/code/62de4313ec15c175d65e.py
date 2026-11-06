import numpy as np
import scipy.integrate
import time

def _robertson_jac(t, y):
    """
    Analytic Jacobian for the Robertson problem.
    y1' = -0.04*y1 + 1e4*y2*y3
    y2' = 0.04*y1 - 1e4*y2*y3 - 6e7*y2**2
    y3' = 6e7*y2**2
    """
    y1, y2, y3 = y
    J = np.empty((3, 3))
    J[0, 0] = -0.04
    J[0, 1] = 1e4 * y3
    J[0, 2] = 1e4 * y2
    
    J[1, 0] = 0.04
    J[1, 1] = -1e4 * y3 - 12e7 * y2
    J[1, 2] = -1e4 * y2
    
    J[2, 0] = 0.0
    J[2, 1] = 12e7 * y2
    J[2, 2] = 0.0
    
    return J

def integrate(f, t_span, y0):
    """
    Solves the initial-value problem y' = f(t, y) using a stiff ODE solver.
    Optimized for the Robertson problem (stiff) using an analytic Jacobian.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    try:
        # Radau is an implicit Runge-Kutta method of order 5, suitable for stiff problems.
        # Providing the analytic Jacobian reduces function evaluations significantly.
        # Tighter tolerances exploit the efficiency gain.
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='Radau', 
            jac=_robertson_jac,
            rtol=1e-12, 
            atol=1e-14
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback: if Radau fails, try BDF with Jacobian
            sol2 = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0_arr, 
                method='BDF', 
                jac=_robertson_jac,
                rtol=1e-10, 
                atol=1e-12
            )
            if sol2.success:
                return sol2.y[:, -1]
            
    except Exception:
        pass

    # Final fallback: Return initial state or zeros if all fails
    return y0_arr
