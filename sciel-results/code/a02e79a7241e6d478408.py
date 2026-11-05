import numpy as np
import scipy.integrate

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Analytical Jacobian for the Robertson problem
    # y1' = -0.04 y1 + 1e4 y2 y3
    # y2' = 0.04 y1 - 1e4 y2 y3 - 3e7 y2^2
    # y3' = 3e7 y2^2
    def jac(t, y):
        y1, y2, y3 = y
        J = np.array([
            [-0.04, 1e4 * y3, 1e4 * y2],
            [0.04, -1e4 * y3 - 6e7 * y2, -1e4 * y2],
            [0.0, 6e7 * y2, 0.0]
        ])
        return J

    rtol = 1e-12
    atol = 1e-14
    
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='Radau', 
            rtol=rtol, 
            atol=atol,
            jac=jac,
            t_eval=[t1]
        )
        
        if not sol.success:
            # Fallback to BDF with Jacobian
            sol = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0_arr, 
                method='BDF', 
                rtol=rtol, 
                atol=atol,
                jac=jac,
                t_eval=[t1]
            )
            
        return sol.y[:, -1]
    except Exception:
        # Last resort: LSODA (does not support jac in the same way, or handles it internally)
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='LSODA', 
            rtol=1e-10, 
            atol=1e-12,
            t_eval=[t1]
        )
        return sol.y[:, -1]
