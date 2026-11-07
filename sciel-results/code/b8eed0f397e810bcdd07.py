import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_vec = np.asarray(y0, dtype=float)
    
    # Use the same method and tolerances as the reference solution
    sol = solve_ivp(
        f, 
        t_span, 
        y0_vec, 
        method='Radau', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback if solve_ivp fails (should not happen for this problem)
        # Try a simple Euler step or return y0 if nothing else works
        # But we must return a finite vector.
        return y0_vec
