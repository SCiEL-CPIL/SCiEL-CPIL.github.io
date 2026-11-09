"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

_START_TIME = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s

def _check_time():
    if time.time() - _START_TIME > _TIME_LIMIT:
        raise TimeoutError("Time limit exceeded")

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use Radau IIA for stiff problems (2D Heat Equation is stiff)
    # Radau is L-stable and handles stiffness well with adaptive steps.
    # rtol/atol chosen to balance accuracy and step count (calls).
    # The score penalizes log(calls), so we want minimal calls for given accuracy.
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14,
            max_step=0.01
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if Radau fails (unlikely for Heat)
            # Try BDF as another implicit method
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-12, 
                atol=1e-14
            )
            if sol.success:
                return sol.y[:, -1]
            # If all fails, return last valid state or y0
            return sol.y[:, -1] if sol.y.size > 0 else y0
    except Exception:
        # In case of timeout or other errors, return best effort
        # Since we can't easily catch partial results from solve_ivp without callbacks,
        # and the problem is fixed, a simple fallback is safer.
        # However, Radau is very robust for this problem.
        return y0

# EVOLVE-BLOCK-END
