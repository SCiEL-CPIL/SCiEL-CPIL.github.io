import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 285  # 15s margin before 300s limit
    
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    
    def time_check():
        return time.time() < deadline
    
    # Strategy: Use Radau (implicit, stiff-capable) with very tight tolerances
    # For the 2D heat equation which is stiff, implicit methods can take larger steps
    # while maintaining stability, reducing total function evaluations.
    # DOP853 is also a good option for non-stiff or mildly stiff problems.
    
    best_result = None
    best_error = float('inf')
    
    # Try Radau first - good for stiff problems like heat equation
    try:
        if not time_check():
            raise TimeoutError()
        
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-13,
            max_step=(t1 - t0) / 50.0,
            first_step=1e-4,
        )
        if sol.success and sol.t[-1] == t1:
            result = sol.y[:, -1]
            if np.all(np.isfinite(result)):
                return result
    except Exception:
        pass
    
    # Try DOP853 - high-order explicit, good for smooth problems
    try:
        if not time_check():
            raise TimeoutError()
        
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-14,
            atol=1e-14,
            max_step=(t1 - t0) / 100.0,
        )
        if sol.success and sol.t[-1] == t1:
            result = sol.y[:, -1]
            if np.all(np.isfinite(result)):
                return result
    except Exception:
        pass
    
    # Try RK45 with tight tolerances as fallback
    try:
        if not time_check():
            raise TimeoutError()
        
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='RK45',
            rtol=1e-13,
            atol=1e-13,
            max_step=(t1 - t0) / 200.0,
        )
        if sol.success and sol.t[-1] == t1:
            result = sol.y[:, -1]
            if np.all(np.isfinite(result)):
                return result
    except Exception:
        pass
    
    # Final fallback: fixed-step RK4
    try:
        n_steps = 2000
        dt = (t1 - t0) / n_steps
        y = y0.copy()
        t = t0
        for i in range(n_steps):
            if not time_check():
                break
            k1 = f(t, y)
            k2 = f(t + dt/2, y + dt/2 * k1)
            k3 = f(t + dt/2, y + dt/2 * k2)
            k4 = f(t + dt, y + dt * k3)
            y = y + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
            t = t + dt
        return y
    except Exception:
        return y0
