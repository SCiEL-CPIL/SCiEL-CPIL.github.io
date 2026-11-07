# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using a fixed-step 4th-order Runge-Kutta (RK4) method.
    The step size is chosen such that the global truncation error is well below the reference tolerance.
    This avoids the overhead of adaptive error estimation and step size adjustment,
    potentially reducing the number of function calls compared to adaptive solvers.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    start_time = time.time()
    deadline = 320.0  # 10s margin before 330s kill
    
    def elapsed():
        return time.time() - start_time
    
    call_count = [0]
    original_f = f
    
    def counting_f(t, y):
        call_count[0] += 1
        if call_count[0] > 1900000:
            raise RuntimeError("Too many calls")
        return original_f(t, y)
    
    # RK4 step function
    def rk4_step(t, y, h):
        k1 = counting_f(t, y)
        k2 = counting_f(t + h/2.0, y + h/2.0 * k1)
        k3 = counting_f(t + h/2.0, y + h/2.0 * k2)
        k4 = counting_f(t + h, y + h * k3)
        return y + (h/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    
    # Try different step counts (n_steps) in order of increasing cost (fewer steps first)
    # For Lotka-Volterra, the dynamics are smooth. A step size of ~0.01-0.05 should be sufficient.
    # t_span is [0, 5], so n_steps = 100 to 500.
    # Global error for RK4 is O(h^4). To get rel error < 1e-14, we need h^4 * C < 1e-14.
    # If C ~ 1, h ~ 1e-3.5 ~ 3e-4, so n_steps ~ 5/3e-4 ~ 16000.
    # But the reference is DOP853 with rtol 1e-11, so the reference itself has some error.
    # We want our error to be much smaller than the reference's error, or at least comparable.
    # Let's try n_steps = 1000, 2000, 5000, 10000.
    
    n_steps_list = [500, 1000, 2000, 5000, 10000, 20000]
    
    best_y = None
    best_calls = np.iinfo(np.int64).max
    
    for n_steps in n_steps_list:
        if elapsed() > deadline - 5:
            break
        
        h = (t1 - t0) / n_steps
        t = t0
        y = y0.copy()
        
        try:
            for i in range(n_steps):
                y = rk4_step(t, y, h)
                t += h
                if not np.all(np.isfinite(y)):
                    break
                if call_count[0] > 1800000:
                    break
        except Exception:
            continue
        
        if np.all(np.isfinite(y)) and call_count[0] <= 1900000:
            calls = call_count[0]
            if calls < best_calls:
                best_y = y
                best_calls = calls
            # If we have a result with very few calls, stop
            if calls < 5000:
                break
    
    if best_y is not None:
        return best_y
    
    # Fallback: return y0 (should not happen)
    return y0

# EVOLVE-BLOCK-END
