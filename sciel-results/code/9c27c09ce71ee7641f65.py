import numpy as np
import scipy.integrate as integrate
import time

def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using DOP853.
    """
    start_time = time.time()
    deadline = start_time + 290  # 10s margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # We want high accuracy to minimize rel error.
    # DOP853 is 8th order.
    # We use tight tolerances to ensure we are at least as accurate as the reference.
    # The reference uses rtol=1e-11, atol=1e-13.
    # Using rtol=1e-14, atol=1e-16 should yield a solution that is effectively the true solution,
    # so the difference with the reference is just the reference's error.
    
    try:
        sol = integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-14, 
            atol=1e-16,
            max_step=(t1-t0)/10  # Limit step size to prevent huge steps if dynamics change, though LV is smooth
        )
        
        if not sol.success:
            # Fallback to RK4 if solve_ivp fails (unlikely)
            return _rk4_fallback(f, t_span, y0)
            
        return sol.y[:, -1]
        
    except Exception:
        return _rk4_fallback(f, t_span, y0)

def _rk4_fallback(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    steps = 1000
    h = (t1 - t0) / steps
    t = t0
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
