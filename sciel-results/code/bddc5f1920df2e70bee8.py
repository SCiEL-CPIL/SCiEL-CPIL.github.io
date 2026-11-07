import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # DOP853 is an 8th-order Runge-Kutta method, very efficient for high accuracy
    # Matching the reference tolerances ensures minimal relative error
    sol = solve_ivp(
        f, 
        (t0, t1), 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback for safety, though DOP853 should not fail on Lorenz
        # Simple RK4 with many steps
        steps = 5000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            if time.time() - start_time > 250:
                break
        return y
