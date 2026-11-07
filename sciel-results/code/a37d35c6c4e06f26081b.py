"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Ensure f is called at least once
    _ = f(t0, y0_arr)
    
    # Use DOP853 with tight tolerances to match reference accuracy
    # Slightly relaxed tolerances to reduce function calls while maintaining high accuracy
    sol = solve_ivp(f, (t0, t1), y0_arr, method='DOP853', rtol=1e-11, atol=1e-13, dense_output=False)
    
    if sol.success and sol.y.shape[1] > 0:
        y_end = sol.y[:, -1]
        if np.all(np.isfinite(y_end)):
            return y_end
    
    # Fallback: RK4 with fixed step size
    n_steps = 10000
    h = (t1 - t0) / n_steps
    t = t0
    y = y0_arr.copy()
    
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    
    return y
# EVOLVE-BLOCK-END
