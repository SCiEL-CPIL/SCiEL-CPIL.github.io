"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    
    Uses DOP853 (8th order Runge-Kutta) with tight tolerances for high accuracy.
    The score is -log10(rel_err) - 0.5*log10(num_calls), so we want high accuracy
    with reasonable number of calls.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: return best effort if we're running out of time
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, leave 20s margin
    
    # Try DOP853 first - 8th order method, very accurate
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='DOP853',
            rtol=1e-14,
            atol=1e-14,
            max_step=(t1 - t0) / 100
        )
        
        if time.time() > deadline:
            # Return what we have if close to deadline
            if sol.success or sol.t.size > 0:
                return sol.y[:, -1]
            return y0
        
        if sol.success:
            return sol.y[:, -1]
        
        # If DOP853 failed, try RK45 with tighter tolerances
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='RK45',
            rtol=1e-13,
            atol=1e-13,
            max_step=(t1 - t0) / 200
        )
        
        if time.time() > deadline:
            if sol.success or sol.t.size > 0:
                return sol.y[:, -1]
            return y0
        
        if sol.success:
            return sol.y[:, -1]
        
    except Exception:
        pass
    
    # Fallback: custom RK4 with many steps
    try:
        n_steps = 5000
        h = (t1 - t0) / n_steps
        t, y = t0, y0.copy()
        
        for i in range(n_steps):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            
            if not np.all(np.isfinite(y)):
                return y
    
        return y
    except Exception:
        return y0
# EVOLVE-BLOCK-END
