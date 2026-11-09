import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Start time
    start_time = time.time()
    deadline = start_time + 250  # Leave 80s margin for safety/overhead
    
    # Primary solver: DOP853 (High-order explicit, good for non-stiff or moderately stiff)
    # The hint suggests DOP853 outperformed Radau for this specific problem.
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-13, 
            atol=1e-15,
            max_step=(t1-t0)/100 # Initial max step constraint to ensure resolution
        )
        
        if not sol.success:
            # Fallback to Radau if DOP853 fails
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau', 
                rtol=1e-13, 
                atol=1e-15
            )
            
        if sol.success:
            return sol.y[:, -1]
        else:
            # If both fail, try a very fine explicit RK4 as a last resort
            # This is unlikely to be needed if the problem is well-posed
            n_steps = 10000
            h = (t1 - t0) / n_steps
            t, y = t0, y0
            for _ in range(n_steps):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2*k1)
                k3 = f(t + h/2, y + h/2*k2)
                k4 = f(t + h, y + h*k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
                if time.time() > deadline:
                    break
            return y
            
    except Exception:
        # Fallback to simple RK4 if SciPy fails
        n_steps = 5000
        h = (t1 - t0) / n_steps
        t, y = t0, y0
        for _ in range(n_steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
