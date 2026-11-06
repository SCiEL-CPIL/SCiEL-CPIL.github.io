import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Strategy: Use LSODA as primary.
    # LSODA automatically switches between Adams (non-stiff) and BDF (stiff).
    # This is a different primary approach than the previous Radau-first strategy.
    # Tight tolerances (1e-12) are used to maximize accuracy score.
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='LSODA', 
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: Radau (Implicit, good for stiff)
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: DOP853 (Explicit, high order, good for smooth)
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: Return initial state (should not happen)
    return y0
