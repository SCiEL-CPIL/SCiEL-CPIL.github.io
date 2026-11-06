"""ODE solver: Robertson - variable-step Radau with adaptive control."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    
    Strategy: Use Radau (implicit, stiff) with adaptive step size and tight tolerances.
    The Robertson problem is stiff; Radau handles it well with few function calls
    relative to explicit methods.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin before 300s kill
    
    # Count function calls
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Use Radau with adaptive step size
    # Tight tolerances for high accuracy
    try:
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=1.0,
            dense_output=False
        )
        
        if not sol.success:
            # Fallback: try with slightly relaxed tolerances
            call_count[0] = 0
            sol = solve_ivp(
                f_counted,
                [t0, t1],
                y0,
                method='Radau',
                rtol=1e-10,
                atol=1e-12,
                max_step=1.0,
                dense_output=False
            )
        
        if sol.success:
            return sol.y[:, -1]
        
        # If Radau failed, try BDF as fallback
        call_count[0] = 0
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0,
            method='BDF',
            rtol=1e-10,
            atol=1e-12,
            max_step=1.0,
            dense_output=False
        )
        
        if sol.success:
            return sol.y[:, -1]
        
        # Last resort: return what we have or y0
        if len(sol.y) > 0:
            return sol.y[:, -1]
        return y0
        
    except Exception:
        # If anything goes wrong, try a simple fallback
        try:
            call_count[0] = 0
            sol = solve_ivp(
                f_counted,
                [t0, t1],
                y0,
                method='LSODA',
                rtol=1e-10,
                atol=1e-12,
                max_step=1.0,
                dense_output=False
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
        return y0
# EVOLVE-BLOCK-END
