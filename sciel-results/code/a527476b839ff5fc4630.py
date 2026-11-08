"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = 290.0  # Safety margin before 300s limit
    
    # Use DOP853 (8th order) with very tight tolerances for high accuracy
    # The convection-diffusion problem is smooth, so explicit high-order methods work well
    
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method='DOP853',
            rtol=1e-14,
            atol=1e-16,
            max_step=(t1 - t0) / 1000,  # Limit step size for accuracy
        )
        
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
        
        # Fallback if DOP853 failed
        # Try with slightly looser tolerances
        sol2 = scipy.integrate.solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 500,
        )
        
        if sol2.success and sol2.y.shape[1] > 0:
            return sol2.y[:, -1]
    
    except Exception:
        pass
    
    # Final fallback: RK4 with many steps
    steps = 2000
    h = (t1 - t0) / steps
    t, y = t0, y0_arr.copy()
    
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
# EVOLVE-BLOCK-END
