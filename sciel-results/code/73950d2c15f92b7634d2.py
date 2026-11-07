"""ODE integrator for the Lotka-Volterra system using DOP853 with adaptive tolerances."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the ODE system using scipy's DOP853 method.
    Uses tight tolerances matching the reference to minimize error and calls.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Start with the same tolerances as the reference
    # DOP853 is 8th order, so it's very efficient
    result = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13
    )
    
    if result.success and len(result.y) > 0:
        return result.y[:, -1]
    
    # If that fails, try with slightly looser tolerances (shouldn't happen)
    result = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='DOP853',
        rtol=1e-10,
        atol=1e-12
    )
    
    if result.success and len(result.y) > 0:
        return result.y[:, -1]
    
    # Fallback: use Radau or BDF for stiff systems
    try:
        result = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12
        )
        if result.success and len(result.y) > 0:
            return result.y[:, -1]
    except:
        pass
    
    # Last resort: simple RK4 with many steps
    h = (t1 - t0) / 10000
    t, y = t0, y0.copy()
    n_steps = int(round((t1 - t0) / h))
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
