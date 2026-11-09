import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Use LSODA which is a hybrid method suitable for stiff and non-stiff problems
    # It often requires fewer function evaluations than purely implicit methods for this problem
    sol = solve_ivp(f, t_span, y0, method='LSODA', rtol=1e-10, atol=1e-12, dense_output=False)
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback to BDF if LSODA fails
        sol = solve_ivp(f, t_span, y0, method='BDF', rtol=1e-10, atol=1e-12, dense_output=False)
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to Radau
            sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12, dense_output=False)
            if sol.success:
                return sol.y[:, -1]
            else:
                return sol.y[:, -1] if sol.y.size > 0 else np.array(y0, dtype=float)
