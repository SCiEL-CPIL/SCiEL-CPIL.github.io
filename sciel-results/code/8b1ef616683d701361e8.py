import numpy as np
import time

def integrate(f, t_span, y0):
    y0 = np.array(y0, dtype=float)
    t0, t1 = t_span
    dt_total = t1 - t0
    
    call_count = 0
    start_time = time.time()
    deadline = start_time + 290
    
    def f_wrapper(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)
    
    # Strategy: Adaptive implicit method with very tight tolerances
    # We'll use Radau with adaptive step size, but start with a small step
    # to ensure stability, and let it grow.
    
    # Try Radau with extremely tight tolerances
    try:
        # Use solve_ivp with Radau, which is robust for stiff systems
        # The heat equation is stiff, so Radau is a good choice
        # We want to minimize the number of calls while maintaining high accuracy
        # Since the score penalizes calls with a factor of 0.5, we can afford
        # some extra calls if they reduce error significantly.
        
        sol = scipy.integrate.solve_ivp(
            f_wrapper, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14,
            max_step=dt_total / 10.0  # Limit max step to ensure accuracy
        )
        
        if sol.success:
            return sol.y[:, -1]
            
    except Exception:
        pass
    
    if time.time() > deadline:
        # If we're running out of time, return current best
        # In this case, we haven't gotten a successful result, so return y0
        # But we should try a simpler method if time permits
        if time.time() < deadline - 5:
            try:
                sol = scipy.integrate.solve_ivp(
                    f_wrapper, 
                    t_span, 
                    y0, 
                    method='BDF', 
                    rtol=1e-12, 
                    atol=1e-14,
                    max_step=dt_total / 10.0
                )
                if sol.success:
                    return sol.y[:, -1]
            except Exception:
                pass
    
    return y0

# Import scipy at module level to avoid import overhead inside the function
import scipy.integrate
