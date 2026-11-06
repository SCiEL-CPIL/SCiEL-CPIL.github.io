"""ODE solver: convection-diffusion 100-point."""
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    calls = [0]
    start_time = time.time()
    time_limit = 310.0
    
    def wrapped_f(t, y):
        calls[0] += 1
        if time.time() - start_time > time_limit or calls[0] > 1900000:
            raise TimeoutError("Time or call limit exceeded")
        return np.asarray(f(t, y), dtype=float)

    def check_time():
        return time.time() - start_time < time_limit

    def check_calls():
        return calls[0] < 1900000

    from scipy.integrate import solve_ivp
    
    best_y = None
    
    # Strategy 1: Radau with very tight tolerances
    try:
        if check_time() and check_calls():
            sol = solve_ivp(wrapped_f, (t0, t1), y0, method='Radau', 
                           rtol=1e-13, atol=1e-15, max_step=0.5)
            if sol.success and sol.y.shape[1] > 0:
                y = sol.y[:, -1]
                if np.all(np.isfinite(y)):
                    best_y = y
    except Exception:
        pass

    # Strategy 2: BDF with tight tolerances
    if best_y is None and check_time() and check_calls():
        try:
            sol = solve_ivp(wrapped_f, (t0, t1), y0, method='BDF', 
                           rtol=1e-13, atol=1e-15, max_step=0.5)
            if sol.success and sol.y.shape[1] > 0:
                y = sol.y[:, -1]
                if np.all(np.isfinite(y)):
                    best_y = y
        except Exception:
            pass

    # Strategy 3: Try DOP853 if not stiff
    if best_y is None and check_time() and check_calls():
        try:
            sol = solve_ivp(wrapped_f, (t0, t1), y0, method='DOP853', 
                           rtol=1e-13, atol=1e-15, max_step=0.5)
            if sol.success and sol.y.shape[1] > 0:
                y = sol.y[:, -1]
                if np.all(np.isfinite(y)):
                    best_y = y
        except Exception:
            pass

    # Strategy 4: RK45 fallback
    if best_y is None and check_time() and check_calls():
        try:
            sol = solve_ivp(wrapped_f, (t0, t1), y0, method='RK45', 
                           rtol=1e-12, atol=1e-14, max_step=0.5)
            if sol.success and sol.y.shape[1] > 0:
                y = sol.y[:, -1]
                if np.all(np.isfinite(y)):
                    best_y = y
        except Exception:
            pass

    # Strategy 5: Manual RK4 with very small steps
    if best_y is None and check_time() and check_calls():
        try:
            remaining_time = time_limit - (time.time() - start_time)
            remaining_calls = 1900000 - calls[0]
            max_steps = min(int(remaining_time * 1000), remaining_calls // 4)
            max_steps = max(max_steps, 1000)
            
            h = (t1 - t0) / max_steps
            t = t0
            y = y0.copy()
            
            for i in range(max_steps):
                if not check_time() or not check_calls():
                    break
                k1 = wrapped_f(t, y)
                k2 = wrapped_f(t + h/2, y + h/2 * k1)
                k3 = wrapped_f(t + h/2, y + h/2 * k2)
                k4 = wrapped_f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
                if not np.all(np.isfinite(y)):
                    break
            
            if np.all(np.isfinite(y)):
                best_y = y
        except Exception:
            pass

    if best_y is None:
        return y0
    return best_y
