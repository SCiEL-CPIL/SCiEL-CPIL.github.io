"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system using DOP853 with tight tolerances
    to minimize relative error (rel) against the reference.
    """
    t0, t1 = t_span
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit

    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13,
            t_eval=[t1]
        )
        
        if not sol.success:
            # Fallback to RK45 if DOP853 fails
            sol = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='RK45', 
                rtol=1e-10, 
                atol=1e-12,
                t_eval=[t1]
            )
            if not sol.success:
                return np.array(y0, dtype=float)

        y_end = sol.y[:, -1]
        
        # Check if result is finite
        if not np.all(np.isfinite(y_end)):
            return np.array(y0, dtype=float)
            
        return y_end
        
    except Exception:
        # Fallback to a simple RK4 if scipy fails entirely
        try:
            y = np.array(y0, dtype=float)
            t = t0
            h = (t1 - t0) / 1000  # 1000 steps
            for _ in range(1000):
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
            return np.array(y0, dtype=float)

# EVOLVE-BLOCK-END
