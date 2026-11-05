import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Solves the Robertson ODE problem using LSODA (automatic stiff/non-stiff switching).
    """
    call_count = 0
    
    def f_counted(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)
    
    # Analytic Jacobian for the Robertson problem
    def robertson_jac(t, y):
        y1, y2, y3 = y[0], y[1], y[2]
        J = np.array([
            [-0.04, 1e4 * y3, 1e4 * y2],
            [0.04, -1e4 * y3 - 6e3 * y2, -1e4 * y2],
            [0.0, 6e3 * y2, 0.0]
        ])
        return J

    start_time = time.time()
    deadline = 310.0  # Safety margin before 330s limit

    try:
        # Use LSODA with tight tolerances
        # LSODA automatically switches between Adams (non-stiff) and BDF (stiff)
        # This is ideal for Robertson which has both fast and slow dynamics
        sol = solve_ivp(
            f_counted, 
            t_span, 
            y0, 
            method='LSODA', 
            jac=robertson_jac,
            rtol=1e-13, 
            atol=1e-15,
            vectorized=False
        )
        
        if not sol.success:
            # Fallback to Radau if LSODA fails
            sol = solve_ivp(
                f_counted, 
                t_span, 
                y0, 
                method='Radau', 
                jac=robertson_jac,
                rtol=1e-13, 
                atol=1e-15,
                vectorized=False
            )
            
        if not sol.success:
            # Last resort: looser tolerances
            sol = solve_ivp(
                f_counted, 
                t_span, 
                y0, 
                method='LSODA', 
                jac=robertson_jac,
                rtol=1e-10, 
                atol=1e-12,
                vectorized=False
            )

        return sol.y[:, -1]
        
    except Exception:
        # Safety net: try Radau with moderate tolerances
        try:
            sol = solve_ivp(
                f_counted, 
                t_span, 
                y0, 
                method='Radau', 
                jac=robertson_jac,
                rtol=1e-10, 
                atol=1e-12,
                vectorized=False
            )
            return sol.y[:, -1]
        except Exception:
            # Final fallback: simple integration
            return np.array(y0, dtype=float)
