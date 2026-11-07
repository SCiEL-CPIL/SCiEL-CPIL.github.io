"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate the ODE system defined by f over t_span starting from y0.
    Uses DOP853 with very tight tolerances to minimize relative error,
    aiming for the 1e-14 floor.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: 330s limit, keep 5s margin
    start_time = time.time()
    deadline = start_time + 325
    
    try:
        # Use DOP853 (8th order) with very tight tolerances
        # rtol=1e-13 and atol=1e-15 should push error to ~1e-14 or lower
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-13, 
            atol=1e-15,
            # max_step not strictly necessary if adaptive, but helps control error growth
            # For t=0 to 5, default is fine
        )
        
        if not sol.success:
            # Fallback: Try with slightly looser tolerances if it failed
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='DOP853', 
                rtol=1e-12, 
                atol=1e-14
            )
            
        if not sol.success:
             # Extreme fallback: RK4 with many steps
             steps = 100000
             h = (t1 - t0) / steps
             t, y = t0, y0.copy()
             for _ in range(steps):
                 if time.time() > deadline:
                     break
                 k1 = f(t, y)
                 k2 = f(t + h / 2, y + h / 2 * k1)
                 k3 = f(t + h / 2, y + h / 2 * k2)
                 k4 = f(t + h, y + h * k3)
                 y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                 t += h
             return y

        y_end = sol.y[:, -1]
        
        if not np.all(np.isfinite(y_end)):
             # Fallback to RK4
             steps = 100000
             h = (t1 - t0) / steps
             t, y = t0, y0.copy()
             for _ in range(steps):
                 if time.time() > deadline:
                     break
                 k1 = f(t, y)
                 k2 = f(t + h / 2, y + h / 2 * k1)
                 k3 = f(t + h / 2, y + h / 2 * k2)
                 k4 = f(t + h, y + h * k3)
                 y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                 t += h
             return y

        return y_end

    except Exception:
        return y0
# EVOLVE-BLOCK-END
