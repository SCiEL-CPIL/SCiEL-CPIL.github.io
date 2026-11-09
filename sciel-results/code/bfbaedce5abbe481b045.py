"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    # Use DOP853 (8th order explicit) for high accuracy with few steps/calls.
    # rtol=1e-12 ensures accuracy better than the reference (Radau rtol=1e-11).
    # This minimizes the number of function calls while keeping rel error low.
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-12, 
        atol=1e-14
    )
    
    if not sol.success:
        # Fallback to Radau if DOP853 fails (unlikely for this problem)
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13
        )
        
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
