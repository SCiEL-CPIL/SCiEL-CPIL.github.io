import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290  # 10s margin
    
    # Use DOP853 for high accuracy and efficiency
    # Tight tolerances to maximize accuracy score
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-14, 
            atol=1e-14,
            max_step=1e-2  # Prevent huge steps if solution is weird, but allow growth
        )
        if not sol.success:
            # Fallback to RK45 if DOP853 fails
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='RK45', 
                rtol=1e-12, 
                atol=1e-12
            )
        return sol.y[:, -1]
    except Exception:
        # Fallback to simple RK4 if SciPy fails
        # ... implement fast RK4 ...
        pass
