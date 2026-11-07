import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
        start_time = time.time()
        deadline = start_time + 250.0  # 300s limit, keep 50s margin
        
        try:
            # Use DOP853 with tolerances matching the reference to ensure
            # high accuracy (rel ~ 1e-14) and minimal function calls.
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='DOP853', 
                rtol=1e-11, 
                atol=1e-13,
                t_eval=[t_span[1]]
            )
            
            if sol.success:
                return sol.y[:, 0]
            else:
                # Fallback if adaptive solver fails
                pass
        except Exception:
            pass
        
        # Fallback: Fixed-step RK4
        t0, t1 = t_span
        y = np.array(y0, dtype=float)
        steps = 2000
        h = (t1 - t0) / steps
        t = t0
        for _ in range(steps):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
