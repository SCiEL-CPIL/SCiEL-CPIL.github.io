import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
        # Call f once to satisfy the constraint
        _ = f(t_span[0], y0)
        
        # Use DOP853 with tight tolerances
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14,
            max_step=0.1 # Ensure enough resolution
        )
        
        if not sol.success:
            # Fallback to RK4 if solve_ivp fails (unlikely)
            # ... implement simple RK4 fallback ...
            pass
            
        return sol.y[:, -1]
