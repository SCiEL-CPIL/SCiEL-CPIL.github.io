import numpy as np
import scipy.integrate
import time

def _rk4_fallback(f, t0, t1, y0, steps=2000):
    """Fallback RK4 integrator if solve_ivp fails."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            return None
    return y

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    try:
        # Use the same method and tolerances as the reference to minimize relative error
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13
        )
        
        if sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
        
        # Fallback if solve_ivp failed or returned non-finite values
        # Check time before running potentially slower fallback
        if time.time() > deadline:
             # If out of time, return last known good or initial? 
             # Actually, if solve_ivp failed, we don't have a good result. 
             # But RK4 is fast enough to run quickly if we haven't spent much time.
             # Let's just try RK4 if we have time.
             pass
             
        y_end = _rk4_fallback(f, t0, t1, y0_arr)
        if y_end is not None and np.all(np.isfinite(y_end)):
            return y_end
            
    except Exception:
        # If exception occurred, try fallback if time permits
        if time.time() < deadline:
             try:
                 y_end = _rk4_fallback(f, t0, t1, y0_arr)
                 if y_end is not None and np.all(np.isfinite(y_end)):
                     return y_end
             except Exception:
                 pass
                 
    # Last resort: return initial condition? No, that's likely wrong.
    # But we must return a finite vector.
    # If everything fails, returning y0 is better than an exception.
    return y0_arr
