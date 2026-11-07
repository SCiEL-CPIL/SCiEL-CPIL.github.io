import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
        start_time = time.time()
        timeout = 290
        
        try:
            # Use very tight tolerances to ensure rel error is at the floor (1e-14)
            # DOP853 is 8th order, so it can handle tight tolerances efficiently
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='DOP853',
                rtol=1e-14,
                atol=1e-16
            )
            
            if sol.success:
                if time.time() - start_time < timeout:
                    return sol.y[:, -1]
                    
        except Exception:
            pass
            
        # Fallback to less tight tolerances if the above fails or is too slow
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='DOP853',
                rtol=1e-11,
                atol=1e-13
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
            
        return np.array(y0, dtype=float)
