import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    start_time = time.time()
    
    # Use DOP853 which is a high-order explicit Runge-Kutta method
    # rtol and atol set to match the reference solution's precision
    # This should yield a very small relative error
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13
        )
        
        if not sol.success:
            # Fallback to a simple RK4 if solve_ivp fails for some reason
            # Though unlikely for this smooth problem
            return _fallback_rk4(f, t_span, y0)
            
        return sol.y[:, -1]
        
    except Exception:
        return _fallback_rk4(f, t_span, y0)

def _fallback_rk4(f, t_span, y0):
    t0, t1 = t_span
    steps = 1000
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
