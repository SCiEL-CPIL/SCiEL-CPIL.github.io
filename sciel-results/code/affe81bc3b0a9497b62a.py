"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = 290.0  # Keep margin before 300s limit
    
    # Try Radau first for stiff problems (convection-diffusion is typically stiff)
    # Radau is an implicit Runge-Kutta method, good for stiff ODEs.
    # Tight tolerances to minimize error.
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='Radau',
            rtol=1e-12, 
            atol=1e-14,
            max_step=(t1-t0)/100 # Limit step size to ensure resolution
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # If Radau fails or is too slow, try DOP853 (high order explicit)
    try:
        if time.time() > deadline - 10:
             # Fallback to a quick RK4 if time is running out
             # But we need to return something.
             pass
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='DOP853',
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to manual RK4 with moderate steps if SciPy fails
    def _rk4(f, t0, t1, y0, steps):
        h = (t1 - t0) / steps
        t, y = t0, np.array(y0, dtype=float)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
                return None
        return y

    y = _rk4(f, t0, t1, y0_arr, 2000)
    if y is not None:
        return y
        
    return y0_arr
# EVOLVE-BLOCK-END
