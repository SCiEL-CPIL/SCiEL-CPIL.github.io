"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Try Radau (implicit, stiff-capable, high order) first.
    # Implicit methods can take larger steps for stiff problems, reducing total f calls.
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            # max_step can be larger for implicit methods
            max_step=0.05 
        )
        
        if time.time() > deadline:
            if sol.t.size > 0:
                return sol.y[:, -1]
            return y0
            
        if sol.success:
            return sol.y[:, -1]
        else:
            if sol.t.size > 0:
                return sol.y[:, -1]
            return y0
            
    except Exception as e:
        # Fallback to DOP853 if Radau fails (e.g., Jacobian issues)
        try:
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='DOP853',
                rtol=1e-12,
                atol=1e-14,
                max_step=0.01
            )
            if time.time() > deadline:
                if sol.t.size > 0:
                    return sol.y[:, -1]
                return y0
                
            if sol.success:
                return sol.y[:, -1]
            else:
                if sol.t.size > 0:
                    return sol.y[:, -1]
                return y0
        except Exception:
            # Last resort: simple RK4 with fixed steps
            try:
                steps = 1000
                h = (t1 - t0) / steps
                t, y = t0, y0.copy()
                for _ in range(steps):
                    k1 = f(t, y)
                    k2 = f(t + h/2, y + h/2 * k1)
                    k3 = f(t + h/2, y + h/2 * k2)
                    k4 = f(t + h, y + h * k3)
                    y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                    t += h
                    if time.time() > deadline:
                        break
                return y
            except Exception:
                return y0
# EVOLVE-BLOCK-END
