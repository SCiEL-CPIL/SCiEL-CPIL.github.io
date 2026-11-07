# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, brentq

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
    
    # The system is:
    # y0' = 1.5*y0 - y0*y1
    # y1' = -3*y1 + y0*y1
    #
    # First integral: V(y0, y1) = 3*y0 - 1.5*ln(y0) + y1 - y1*ln(y1) = const
    # Equilibrium at (3, 1.5)
    #
    # We can use the first integral to find the trajectory, but we still need
    # to integrate the ODE to get the time evolution. However, we can use the
    # fact that the solution is periodic and use a very accurate integrator
    # with minimal steps.
    
    # Strategy: Use DOP853 with very tight tolerances and a single pass.
    # The key insight is that DOP853 with rtol=1e-13, atol=1e-15 should give
    # rel < 1e-14 with relatively few calls.
    
    best_y = None
    best_calls = np.iinfo(np.int64).max
    
    # Try a single high-accuracy DOP853 run
    configs = [
        (1e-13, 1e-15, 0.5),
        (1e-14, 1e-16, 0.5),
        (1e-13, 1e-15, 1.0),
    ]
    
    for rtol, atol, max_step in configs:
        if elapsed() > deadline - 30:
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
    
    # Fallback: RK4 with many steps
    def rk4_step(f, t, y, h):
        k1 = f(t, y)
        k2 = f(t + h/2.0, y + h/2.0 * k1)
        k3 = f(t + h/2.0, y + h/2.0 * k2)
        k4 = f(t + h, y + h * k3)
        return y + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
    
    for n_steps in [5000, 10000, 20000]:
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
    
    return y0

# EVOLVE-BLOCK-END
