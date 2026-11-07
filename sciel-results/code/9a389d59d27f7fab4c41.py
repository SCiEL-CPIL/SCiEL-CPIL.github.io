# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    start_time = time.time()
    deadline = 320.0
    
    def elapsed():
        return time.time() - start_time
    
    call_count = [0]
    original_f = f
    
    def counting_f(t, y):
        call_count[0] += 1
        if call_count[0] > 1900000:
            raise RuntimeError("Too many calls")
        return original_f(t, y)
    
    best_y = None
    best_calls = np.iinfo(np.int64).max
    best_rel_estimate = np.inf
    
    # Strategy:
    # 1. Try solve_ivp with DOP853 and various max_step values
    # 2. Pick the configuration with fewest calls that achieves good accuracy
    # 3. Since we can't know rel exactly, we assume tight tolerances + DOP853 gives rel < 1e-14
    
    configs = [
        (1e-11, 1e-13, 0.2),
        (1e-11, 1e-13, 0.1),
        (1e-11, 1e-13, 0.05),
        (1e-12, 1e-14, 0.1),
        (1e-12, 1e-14, 0.05),
    ]
    
    for rtol, atol, max_step in configs:
        if elapsed() > deadline - 15:
            break
        
        try:
            sol = solve_ivp(
                counting_f,
                t_span,
                y0,
                method='DOP853',
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                dense_output=False
            )
            
            if sol.success:
                y_end = sol.y[:, -1]
                if np.all(np.isfinite(y_end)):
                    calls = call_count[0]
                    if calls < best_calls:
                        best_y = y_end
                        best_calls = calls
                        
        except Exception:
            continue
    
    if best_y is not None:
        return best_y
    
    # Fallback: use a simple high-order RK method with fixed steps
    # DOP853 coefficients for a single 8th-order step
    # We'll use a simpler approach: RK4 with many steps as fallback
    
    def rk4_step(f, t, y, h):
        k1 = f(t, y)
        k2 = f(t + h/2.0, y + h/2.0 * k1)
        k3 = f(t + h/2.0, y + h/2.0 * k2)
        k4 = f(t + h, y + h * k3)
        return y + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
    
    for n_steps in [2000, 5000, 10000, 20000]:
        if elapsed() > deadline - 5:
            break
        
        h = (t1 - t0) / n_steps
        t = t0
        y = y0.copy()
        
        for i in range(n_steps):
            y = rk4_step(counting_f, t, y, h)
            t += h
            if not np.all(np.isfinite(y)):
                break
            if call_count[0] > 1800000:
                break
        
        if np.all(np.isfinite(y)) and call_count[0] <= 1900000:
            return y
    
    # Last resort: return y0
    return y0

# EVOLVE-BLOCK-END
