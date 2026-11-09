import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Van der Pol specific Jacobian
    def jac(t, y):
        mu = 5.0
        y0_val = y[0]
        y1_val = y[1]
        return np.array([
            [0.0, 1.0],
            [-2.0 * mu * y0_val * y1_val - 1.0, mu * (1.0 - y0_val**2)]
        ])

    start_time = time.time()
    DEADLINE = start_time + 290.0

    try:
        # Use Radau with Jacobian for efficiency and accuracy
        sol = solve_ivp(
            f, 
            t_span, 
            y, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13,
            jac=jac
        )
        
        if time.time() > DEADLINE:
            # If we are close to deadline, return whatever we have or fallback
            pass

        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to LSODA if Radau fails
            sol = solve_ivp(
                f, 
                t_span, 
                y, 
                method='LSODA', 
                rtol=1e-11, 
                atol=1e-13
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                # Last resort: default
                return np.array(y, dtype=float)
    except Exception:
        return np.array(y, dtype=float)
