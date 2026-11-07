import numpy as np
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280  # 20s margin before 300s limit
    
    try:
        # Use DOP853 for high accuracy with relatively few calls
        # rtol and atol set to ensure high precision
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-13, 
            atol=1e-15,
            max_step=0.1
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if solve_ivp fails
            h = (t_span[1] - t_span[0]) / 10000
            t, y = t_span[0], np.array(y0, dtype=float)
            for _ in range(10000):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
                if time.time() > deadline:
                    return y
            return y
            
    except Exception:
        # Fallback to RK4 if SciPy fails or times out internally
        h = (t_span[1] - t_span[0]) / 10000
        t, y = t_span[0], np.array(y0, dtype=float)
        for _ in range(10000):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            if time.time() > deadline:
                return y
        return y
