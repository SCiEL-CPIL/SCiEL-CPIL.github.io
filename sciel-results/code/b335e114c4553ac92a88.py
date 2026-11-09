"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    ODE integrator for the Lorenz system.
    
    Uses scipy's DOP853 with tolerances matching the reference solution.
    We add a first_step hint to potentially reduce initial exploration calls.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: 300s limit, keep 10s margin
    deadline = time.time() + 290
    
    try:
        # Use the same parameters as the reference to minimize relative error.
        # Add first_step to help the adaptive solver choose a reasonable initial step,
        # potentially reducing the number of function calls during the initial phase.
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            first_step=0.1
        )
        
        if sol.success:
            result = sol.y[:, -1]
            if np.all(np.isfinite(result)) and result.shape == y0.shape:
                return result
    except Exception:
        pass
    
    # Fallback: Simple RK4 if solve_ivp fails
    n_steps = 10000
    h = (t1 - t0) / n_steps
    t, y = t0, y0.copy()
    for _ in range(n_steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            return y0.copy()
            
    return y
# EVOLVE-BLOCK-END
