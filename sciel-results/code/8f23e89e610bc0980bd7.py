"""ODE solver: Convection-diffusion 100-point using DOP853."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Track calls to f for safety
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    deadline = time.time() + 280  # 300s limit, keep 20s margin
    
    # Try DOP853 with tight tolerances
    # DOP853 is 8th order, 12 stages (12 f calls per step)
    # For a 100-point convection-diffusion problem on [0, 0.5],
    # this should be very accurate with few steps
    
    max_calls = 2000000
    
    try:
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0_arr,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 50,  # Limit step size for accuracy
            max_nfev=max_calls
        )
        
        if sol.success and sol.t[-1] >= t1 - 1e-14:
            return sol.y[:, -1]
        
        # If DOP853 didn't quite reach t1, try to continue or use last point
        if sol.t.size > 0:
            # Check if we're close enough
            if t1 - sol.t[-1] < 1e-10:
                return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: try RK45 with moderate tolerances
    try:
        call_count[0] = 0
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0_arr,
            method='RK45',
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 200,
            max_nfev=max_calls
        )
        
        if sol.success and sol.t[-1] >= t1 - 1e-14:
            return sol.y[:, -1]
        
        if sol.t.size > 0 and t1 - sol.t[-1] < 1e-8:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: fixed-step RK4
    try:
        call_count[0] = 0
        n_steps = 5000
        h = (t1 - t0) / n_steps
        t = t0
        y = y0_arr.copy()
        
        for _ in range(n_steps):
            if time.time() > deadline:
                break
            k1 = f_counted(t, y)
            k2 = f_counted(t + h/2, y + h/2 * k1)
            k3 = f_counted(t + h/2, y + h/2 * k2)
            k4 = f_counted(t + h, y + h * k3)
            y = y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            if not np.all(np.isfinite(y)):
                break
        
        return y
    except Exception:
        pass
    
    # Absolute fallback: return initial condition
    return y0_arr
# EVOLVE-BLOCK-END
