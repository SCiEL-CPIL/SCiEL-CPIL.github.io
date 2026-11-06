import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use a high-order implicit solver for stiff problems (heat equation is stiff)
    # Radau is a good choice for stiff ODEs.
    # Tight tolerances to minimize error.
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-10, 
        atol=1e-12,
        max_step=(t1 - t0) / 10.0  # Limit step size to ensure accuracy, though Radau adapts
    )
    
    if not sol.success:
        # Fallback to BDF if Radau fails
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-10, 
            atol=1e-12
        )
        
    return sol.y[:, -1]
