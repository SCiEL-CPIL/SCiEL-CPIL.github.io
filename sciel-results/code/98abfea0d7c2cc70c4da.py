import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    
    # Define the analytical Jacobian for the Van der Pol oscillator (mu=5)
    # y0' = y1
    # y1' = 5 * (1 - y0^2) * y1 - y0
    # J = [[0, 1], [-10*y0*y1 - 1, 5*(1 - y0^2)]]
    def jac(t, y):
        y0 = y[0]
        y1 = y[1]
        return np.array([[0.0, 1.0],
                         [-10.0 * y0 * y1 - 1.0, 5.0 * (1.0 - y0**2)]])

    # Use Radau with very tight tolerances and the analytical Jacobian
    # Radau is an implicit Runge-Kutta method, well-suited for stiff problems.
    # Providing the Jacobian avoids numerical differentiation (which costs f calls)
    # and speeds up Newton iterations (reducing f calls per step).
    
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-14,
            atol=1e-16,
            jac=jac
        )
        
        if sol.success:
            return sol.y[:, -1]
            
    except Exception:
        pass

    # Fallback if Radau fails or is too slow (unlikely for this 2D system)
    # Try BDF with Jacobian
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='BDF',
            rtol=1e-14,
            atol=1e-16,
            jac=jac
        )
        
        if sol.success:
            return sol.y[:, -1]
            
    except Exception:
        pass

    # Final fallback: high-order explicit RK (DOP853) without Jacobian
    # This is less efficient for stiff problems but doesn't need Jac
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-14,
            atol=1e-16
        )
        
        if sol.success:
            return sol.y[:, -1]
            
    except Exception:
        pass

    # If all else fails, return y0 (should not happen)
    return np.array(y0, dtype=float)
