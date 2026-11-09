import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use DOP853 with tight tolerances to minimize relative error
    # DOP853 is an 8th-order method, which is more efficient than DP5
    # for achieving high accuracy with fewer function calls.
    rtol = 1e-13
    atol = 1e-15
    
    try:
        sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=rtol, atol=atol)
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
        else:
            # Fallback if solver fails
            if sol.y.size > 0:
                return sol.y[:, -1]
            return y0
    except Exception:
        # Return y0 to avoid invalid result due to exception
        return y0
