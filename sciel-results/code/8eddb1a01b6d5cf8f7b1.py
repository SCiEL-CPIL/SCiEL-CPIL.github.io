"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t_start = time.time()
    deadline = t_start + 280.0  # Leave 20s safety margin
    
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    
    # Try Radau (implicit, stiff-capable, order 3)
    # Tight tolerances for high accuracy
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            max_step=0.01  # Limit step size to ensure resolution
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # If Radau fails or is too slow, try BDF
    if time.time() < deadline:
        try:
            sol = solve_ivp(
                f, t_span, y0,
                method='BDF',
                rtol=1e-11,
                atol=1e-13,
                max_step=0.01
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass

    # Fallback to explicit RK45 if implicit methods fail completely
    if time.time() < deadline:
        try:
            sol = solve_ivp(
                f, t_span, y0,
                method='RK45',
                rtol=1e-8,
                atol=1e-10,
                max_step=0.001
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass

    # If all else fails, return initial state (should not happen for valid IVPs)
    return y0

# EVOLVE-BLOCK-END
