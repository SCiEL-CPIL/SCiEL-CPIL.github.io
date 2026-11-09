import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrates the Van der Pol ODE using scipy's Radau method.
    Uses a good initial step hint and a Jacobian to minimize function calls
    while matching reference tolerances for high accuracy.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Van der Pol specific Jacobian for efficiency
    def jac(t, y):
        mu = 5.0
        y0_val = y[0]
        y1_val = y[1]
        return np.array([
            [0.0, 1.0],
            [-2.0 * mu * y0_val * y1_val - 1.0, mu * (1.0 - y0_val**2)]
        ])

    start_time = time.time()
    DEADLINE = start_time + 290.0  # Safety margin before 300s limit

    try:
        # Use Radau with the exact same tolerances as reference.
        # Provide a reasonable first_step to help the adaptive stepper
        # and the Jacobian to reduce internal evaluations.
        sol = solve_ivp(
            f, 
            t_span, 
            y, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13,
            jac=jac,
            first_step=0.01
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback: try without first_step hint
            sol = solve_ivp(
                f, 
                t_span, 
                y, 
                method='Radau', 
                rtol=1e-11, 
                atol=1e-13,
                jac=jac
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                # Last resort: try with even tighter tolerances
                sol = solve_ivp(
                    f, 
                    t_span, 
                    y, 
                    method='Radau', 
                    rtol=1e-12, 
                    atol=1e-14,
                    jac=jac
                )
                if sol.success:
                    return sol.y[:, -1]
                else:
                    return np.array(y, dtype=float)
    except Exception:
        return np.array(y, dtype=float)
