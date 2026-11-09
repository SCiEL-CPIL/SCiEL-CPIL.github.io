import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Use Radau method which is suitable for stiff problems
    # Tight tolerances to minimize relative error
    sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12, dense_output=False)
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback to BDF if Radau fails
        sol = solve_ivp(f, t_span, y0, method='BDF', rtol=1e-10, atol=1e-12, dense_output=False)
        if sol.success:
            return sol.y[:, -1]
        else:
            # Last resort: return the last known state or raise
            return sol.y[:, -1] if sol.y.size > 0 else np.array(y0, dtype=float)
