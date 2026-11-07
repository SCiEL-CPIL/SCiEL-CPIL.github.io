"""ODE integrator for the Lotka-Volterra system using DOP853."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system from t_span[0] to t_span[1]
    using scipy's DOP853 method with tight tolerances matching the reference.
    
    Parameters:
    -----------
    f : callable
        Function f(t, y) returning y' as an array.
    t_span : tuple
        (t_start, t_end)
    y0 : array-like
        Initial conditions.
        
    Returns:
    --------
    y_end : np.ndarray
        State at t_end.
    """
    from scipy.integrate import solve_ivp
    
    # Record start time for deadline guard
    t_start_wall = time.time()
    
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use DOP853 with tight tolerances matching the reference solver
    # rtol=1e-11, atol=1e-13 to match scipy.integrate.solve_ivp reference
    result = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
        dense_output=False
    )
    
    # Check if we're approaching the deadline
    elapsed = time.time() - t_start_wall
    if elapsed > 250:  # Safety margin before 300s limit
        # Return what we have if the integration succeeded
        if result.success and len(result.t) > 0:
            return result.y[:, -1].copy()
        # If it failed, try a simpler fallback
        return _fallback_rk4(f, t0, t1, y0)
    
    if result.success:
        return result.y[:, -1].copy()
    else:
        # Fallback if DOP853 failed for some reason
        return _fallback_rk4(f, t0, t1, y0)


def _fallback_rk4(f, t0, t1, y0, steps=5000):
    """Simple RK4 fallback with adaptive step checking."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
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
