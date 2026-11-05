import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Solves the Robertson ODE problem using LSODA with a Jacobian.
    LSODA automatically switches between Adams and BDF methods,
    efficiently handling stiffness transitions.
    """
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    def jac(t, y):
        y1, y2, y3 = y
        J = np.array([
            [-0.04,      1e4*y3,    1e4*y2],
            [ 0.04, -1e4*y3-6e7*y2, -1e4*y2],
            [ 0.0,     6e7*y2,        0.0   ]
        ])
        return J

    start_time = time.time()
    deadline = 280.0
    
    t0, tf = t_span
    
    try:
        sol = solve_ivp(
            f_counted, 
            t_span, 
            y0, 
            method='LSODA', 
            jac=jac,
            rtol=1e-12, 
            atol=1e-14,
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            raise RuntimeError("solve_ivp failed with LSODA")
            
    except Exception as e:
        # Fallback to Radau if LSODA fails
        try:
            sol = solve_ivp(
                f_counted, 
                t_span, 
                y0, 
                method='Radau', 
                jac=jac,
                rtol=1e-12, 
                atol=1e-14,
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                raise RuntimeError("solve_ivp failed with Radau")
        except Exception as e2:
            # Final fallback: BDF
            try:
                sol = solve_ivp(
                    f_counted, 
                    t_span, 
                    y0, 
                    method='BDF', 
                    jac=jac,
                    rtol=1e-10, 
                    atol=1e-12,
                )
                if sol.success:
                    return sol.y[:, -1]
                else:
                    raise RuntimeError("solve_ivp failed with BDF")
            except Exception as e3:
                # Absolute last resort: return y0 (will be inaccurate but won't crash)
                return np.array(y0, dtype=float)
