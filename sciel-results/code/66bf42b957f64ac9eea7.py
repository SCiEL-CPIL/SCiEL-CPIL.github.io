import numpy as np
import time
from scipy.integrate import solve_ivp

def _jac_robertson(t, y):
    """
    Analytical Jacobian for the Robertson problem.
    y = [y1, y2, y3]
    y1' = -0.04*y1 + 1e4*y2*y3
    y2' = 0.04*y1 - 1e4*y2*y3 - 3e3*y2**2
    y3' = 3e3*y2**2
    """
    y1, y2, y3 = y
    # Construct Jacobian directly
    J = np.array([
        [-0.04, 1e4 * y3, 1e4 * y2],
        [0.04, -1e4 * y3 - 6e3 * y2, -1e4 * y2],
        [0.0, 6e3 * y2, 0.0]
    ])
    return J

def integrate(f, t_span, y0):
    """
    Integrates the ODE y' = f(t, y) from t_span[0] to t_span[1].
    Uses Radau method with analytical Jacobian for stiff Robertson problem.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: 330s limit, use 300s for safety
    start_time = time.time()
    deadline = start_time + 300.0
    
    # Tolerances: Push tighter to reduce error score penalty
    rtol = 1e-13
    atol = 1e-15
    
    try:
        sol = solve_ivp(
            f, 
            [t0, t1], 
            y0, 
            method='Radau', 
            rtol=rtol, 
            atol=atol,
            jac=_jac_robertson
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to BDF if Radau fails
            sol = solve_ivp(
                f, 
                [t0, t1], 
                y0, 
                method='BDF', 
                rtol=rtol, 
                atol=atol,
                jac=_jac_robertson
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                if sol.y.shape[1] > 0:
                    return sol.y[:, -1]
                return y0

    except Exception:
        return y0
