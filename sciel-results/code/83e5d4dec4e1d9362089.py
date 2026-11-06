"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin before 330s kill
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Strategy: Use DOP853 (8th order) with very tight tolerances for high accuracy
    # The score rewards accuracy (-log10(relative_error)) and penalizes evaluations
    # (-0.5*log10(calls)). DOP853 is efficient for non-stiff problems.
    # For 100-point convection-diffusion, it's typically non-stiff or mildly stiff.
    # We use rtol=1e-12, atol=1e-14 to push accuracy near machine precision limits
    # while keeping step count reasonable due to high order.
    
    # First attempt: DOP853 with tight tolerances
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.01
        )
        
        if time.time() > deadline:
            if sol.t.size > 0:
                return sol.y[:, -1]
            return y0
            
        if sol.success:
            return sol.y[:, -1]
        else:
            # If solver failed, return last computed state
            if sol.t.size > 0:
                return sol.y[:, -1]
            return y0
            
    except Exception:
        pass
    
    # Second attempt: Try RK45 with tighter tolerances if DOP853 failed
    try:
        if time.time() < deadline:
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='RK45',
                rtol=1e-11,
                atol=1e-13,
                max_step=0.005
            )
            
            if time.time() > deadline:
                if sol.t.size > 0:
                    return sol.y[:, -1]
                return y0
                
            if sol.success:
                return sol.y[:, -1]
            elif sol.t.size > 0:
                return sol.y[:, -1]
            else:
                return y0
        else:
            return y0
    except Exception:
        pass
    
    # Third attempt: Try Radau for stiff problems with optimized parameters
    try:
        if time.time() < deadline:
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
                max_step=0.1
            )
            
            if time.time() > deadline:
                if sol.t.size > 0:
                    return sol.y[:, -1]
                return y0
                
            if sol.success:
                return sol.y[:, -1]
            elif sol.t.size > 0:
                return sol.y[:, -1]
            else:
                return y0
        else:
            return y0
    except Exception:
        pass
    
    # Last resort: return initial condition
    return y0
# EVOLVE-BLOCK-END
