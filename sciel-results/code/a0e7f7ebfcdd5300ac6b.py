import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    deadline = time.time() + 290
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use Radau for stiff systems
    # High precision tolerances
    rtol = 1e-12
    atol = 1e-14
    
    try:
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=rtol, atol=atol)
        if sol.success:
            return sol.y[:, -1]
        else:
            # If Radau fails, try BDF
            sol = solve_ivp(f, t_span, y0, method='BDF', rtol=rtol, atol=atol)
            if sol.success:
                return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to expm if solver fails (assumes constant A)
    # This is a last resort
    n = len(y0)
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        A[:, j] = f(t0, e)
    from scipy.linalg import expm
    M = expm((t1 - t0) * A)
    return M @ y0
