import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t_start, t_end = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use DOP853 with tight tolerances to match reference and minimize error
    # DOP853 is 8th order, highly efficient for smooth problems like Lorenz
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13,
        max_step=0.01 # Optional: limit max step to ensure resolution, though adaptive usually handles it
    )
    
    if not sol.success:
        raise ValueError("Integration failed: " + sol.message)
        
    return sol.y[:, -1]
