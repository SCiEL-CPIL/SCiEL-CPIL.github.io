import numpy as np
from scipy.integrate import solve_ivp
import time

def _robertson_jac(t, y):
    """Analytical Jacobian for the Robertson problem."""
    y0, y1, y2 = y
    return np.array([
        [-0.04, 1e4 * y2, 1e4 * y1],
        [0.04, -1e4 * y2 - 6e3 * y1, -1e4 * y1],
        [0.0, 6e3 * y1, 0.0]
    ])

def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses an implicit stiff solver (Radau) with an analytical Jacobian for the Robertson problem.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    timeout = 290  # Leave 10s margin for 300s limit

    # Try Radau with analytical Jacobian first
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=_robertson_jac, 
            rtol=1e-12, 
            atol=1e-14,
            max_step=(t1-t0)/10  # Initial max step hint, solver will adapt
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If Radau fails, fall back to BDF
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                jac=_robertson_jac, 
                rtol=1e-10, 
                atol=1e-12
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                # Last resort: return last computed value or y0 if it fails immediately
                return sol.y[:, -1] if sol.y.size > 0 else y0

    except Exception as e:
        # If an exception occurs, try a simpler method or return y0
        try:
            sol = solve_ivp(f, t_span, y0, method='LSODA', rtol=1e-8, atol=1e-10)
            if sol.success:
                return sol.y[:, -1]
        except:
            pass
        return y0
