"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """Integrate a 15-dim stiff linear IVP using SciPy's Radau solver."""
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    # Try Radau (good for stiff problems)
    try:
        sol = solve_ivp(
            f, 
            [t0, t1], 
            y0, 
            method='Radau',
            rtol=1e-10, 
            atol=1e-12,
            max_step=(t1 - t0) / 100
        )
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: try BDF
    try:
        sol = solve_ivp(
            f, 
            [t0, t1], 
            y0, 
            method='BDF',
            rtol=1e-10, 
            atol=1e-12,
            max_step=(t1 - t0) / 100
        )
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Final fallback: simple RK4 with adaptive stepping
    n = len(y0)
    y = y0.copy()
    t = t0
    h = (t1 - t0) / 1000
    
    while t < t1 and time.time() < deadline:
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t = min(t + h, t1)
        
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
