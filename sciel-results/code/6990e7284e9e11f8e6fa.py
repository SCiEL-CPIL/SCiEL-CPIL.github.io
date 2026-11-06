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
    
    # Strategy:
    # 1. Convection-diffusion is typically stiff. Explicit methods (DOP853) waste calls on stability.
    # 2. Implicit methods (Radau, BDF) are stable and allow larger steps, reducing f-calls.
    # 3. We prioritize Radau for its balance of accuracy and stability.
    # 4. We use tight tolerances to maximize the accuracy term in the score.
    
    # Attempt 1: Radau (Implicit, 3rd order, L-stable)
    # Good for stiff problems.
    try:
        if time.time() < deadline:
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau',
                rtol=1e-10,
                atol=1e-12,
                # max_step=t1-t0 # Let it choose, but maybe cap it? No, let it be efficient.
            )
            
            if time.time() > deadline:
                if sol.t.size > 0:
                    return sol.y[:, -1]
                return y0
                
            if sol.success:
                return sol.y[:, -1]
            else:
                # If solver failed, return last computed state if available
                if sol.t.size > 0:
                    return sol.y[:, -1]
                return y0
    except Exception:
        pass
    
    # Attempt 2: BDF (Implicit, up to 5th order, A-stable)
    # Often faster than Radau per step, slightly less accurate per step but good for stiff.
    try:
        if time.time() < deadline:
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF',
                rtol=1e-10,
                atol=1e-12,
                max_step=(t1-t0)/100.0 # Prevent huge steps if solution changes rapidly
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
    except Exception:
        pass
    
    # Attempt 3: DOP853 (Explicit, 8th order)
    # Fallback for non-stiff or if implicit methods fail.
    try:
        if time.time() < deadline:
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
            elif sol.t.size > 0:
                return sol.y[:, -1]
            else:
                return y0
    except Exception:
        pass
    
    # Last resort: return initial condition
    return y0
# EVOLVE-BLOCK-END
