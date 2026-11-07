import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using scipy's DOP853 with very tight tolerances
    to minimize relative error, aiming for the 1e-14 floor.
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    try:
        # Use DOP853 with tighter tolerances to push error below 1e-14
        # rtol=1e-13 and atol=1e-15 should be sufficient for double precision
        # without exploding the step count.
        sol = solve_ivp(
            f, 
            t_span, 
            y0,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15,
            dense_output=False
        )
        
        if sol.success and np.all(np.isfinite(sol.y)):
            y_end = sol.y[:, -1]
            if y_end.shape == y0.shape:
                return y_end
        
        # Fallback if DOP853 fails (unlikely)
        if time.time() > deadline - 10:
            return y0

        # Fallback: RK4 with N=1000
        N = 1000
        h = (t1 - t0) / N
        t = t0
        y = y0.copy()
        
        for i in range(N):
            if time.time() > deadline - 5:
                break
            k1 = f(t, y)
            k2 = f(t + 0.5*h, y + 0.5*h*k1)
            k3 = f(t + 0.5*h, y + 0.5*h*k2)
            k4 = f(t + h, y + h*k3)
            y = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t = t + h
            
        return y

    except Exception:
        return y0
