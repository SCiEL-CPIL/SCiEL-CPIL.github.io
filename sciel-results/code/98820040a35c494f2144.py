import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t_start = time.time()
    deadline = t_start + 280  # 280s limit to be safe
    
    # Use DOP853 for high efficiency and accuracy
    # Tolerances set tight to beat reference
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    if not sol.success:
        # Fallback to a robust fixed-step RK45 or similar if adaptive fails
        # But for this problem, it shouldn't fail.
        # Let's try Radau as a stiff fallback just in case
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13
        )
        
    if not sol.success:
        # Final fallback: manual RK4 with many steps (like seed but maybe fewer if we trust it)
        # Or just return last known state? No, must be finite.
        # Let's stick with the seed's logic as a last resort if scipy fails completely
        t0, t1 = t_span
        y = np.array(y0, dtype=float)
        steps = 2000
        h = (t1 - t0) / steps
        t = t0
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            if time.time() > deadline:
                break
        return y
        
    return sol.y[:, -1]
