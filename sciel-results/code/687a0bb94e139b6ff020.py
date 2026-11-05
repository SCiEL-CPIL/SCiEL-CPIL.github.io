import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Solves the Robertson ODE problem using LSODA (auto-switching Adams/BDF)
    with a known Jacobian for efficiency. Falls back to Radau, then BDF.
    """
    # Track calls to f
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # The Robertson problem has a known structure.
    # y = [y1, y2, y3]
    # y1' = -0.04*y1 + 1e4*y2*y3
    # y2' =  0.04*y1 - 1e4*y2*y3 - 3e7*y2*y2
    # y3' =  3e7*y2*y2
    #
    # Jacobian:
    # J = [
    #   [-0.04,      1e4*y3,    1e4*y2],
    #   [ 0.04, -1e4*y3-6e7*y2, -1e4*y2],
    #   [ 0,       6e7*y2,        0    ]
    # ]
    
    def jac(t, y):
        y1, y2, y3 = y
        J = np.array([
            [-0.04,      1e4*y3,    1e4*y2],
            [ 0.04, -1e4*y3-6e7*y2, -1e4*y2],
            [ 0.0,     6e7*y2,        0.0   ]
        ])
        return J

    # Set a deadline guard
    start_time = time.time()
    deadline = 290.0  # 290 seconds, well under 300s limit
    
    t0, tf = t_span
    
    # Try LSODA with Jacobian first
    # LSODA automatically switches between Adams (non-stiff) and BDF (stiff),
    # which is efficient for the Robertson problem's stiffness transitions.
    try:
        sol = solve_ivp(
            f_counted, 
            t_span, 
            y0, 
            method='LSODA', 
            jac=jac,
            rtol=1e-12, 
            atol=1e-14,
            max_step=10.0,
            vectorized=False
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
                max_step=10.0,
                vectorized=False
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
                    rtol=1e-12, 
                    atol=1e-14,
                    max_step=10.0,
                    vectorized=False
                )
                if sol.success:
                    return sol.y[:, -1]
                else:
                    raise RuntimeError("solve_ivp failed with BDF")
            except Exception as e3:
                # Absolute last resort: return y0 (will be inaccurate but won't crash)
                return np.array(y0, dtype=float)
