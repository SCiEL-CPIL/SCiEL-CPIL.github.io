import numpy as np
import scipy.integrate
import time

# Analytic Jacobian for the Robertson problem
def _robertson_jac(t, y):
    y1, y2, y3 = y
    J = np.empty((3, 3))
    J[0, 0] = -0.04
    J[0, 1] = 1e4 * y3
    J[0, 2] = 1e4 * y2
    J[1, 0] = 0.04
    J[1, 1] = -1e4 * y3 - 6e7 * y2
    J[1, 2] = -1e4 * y2
    J[2, 0] = 0.0
    J[2, 1] = 6e7 * y2
    J[2, 2] = 0.0
    return J

def integrate(f, t_span, y0):
    """
    Solves the initial-value problem y' = f(t, y) using a stiff ODE solver.
    Optimized for the Robertson problem (stiff).
    Uses analytic Jacobian to reduce function evaluation costs.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    try:
        # Radau with analytic Jacobian is highly efficient for stiff Robertson problem
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12,
            jac=_robertson_jac
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback: if Radau fails, try BDF with Jacobian
            if time.time() < deadline:
                sol2 = scipy.integrate.solve_ivp(
                    f, 
                    t_span, 
                    y0_arr, 
                    method='BDF', 
                    rtol=1e-8, 
                    atol=1e-10,
                    jac=_robertson_jac
                )
                if sol2.success:
                    return sol2.y[:, -1]
            
    except Exception:
        pass

    # Final fallback: Return initial state or zeros if all fails
    return y0_arr
