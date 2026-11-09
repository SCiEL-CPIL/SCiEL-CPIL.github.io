"""Improved ODE integrator using SciPy's DOP853 with structural error control."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrates the Lotka-Volterra system from t_span[0] to t_span[1].
    Returns the state at the end of t_span.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin within 300s limit
    
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Handle zero interval
    if abs(t1 - t0) < 1e-15:
        return y0_arr.copy()
    
    # Use DOP853 with very tight tolerances
    # The key insight: DOP853 is 8th order, so it's very efficient.
    # We want to minimize calls while maximizing accuracy.
    # rtol=1e-12, atol=1e-14 should be sufficient for rel ~ 1e-14 or better.
    
    try:
        sol = solve_ivp(
            f, 
            (t0, t1), 
            y0_arr, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14
        )
        
        if not sol.success:
            return _fallback_rk4(f, t_span, y0_arr, deadline)
        
        y_end = sol.y[:, -1]
        
        # Check if finite
        if not np.all(np.isfinite(y_end)):
            return _fallback_rk4(f, t_span, y0_arr, deadline)
            
        return y_end
        
    except Exception:
        # If solve_ivp throws an exception, fall back to RK4
        return _fallback_rk4(f, t_span, y0_arr, deadline)

def _fallback_rk4(f, t_span, y0, deadline=None):
    """
    Fallback fixed-step RK4 integrator.
    Used only if the adaptive solver fails or produces non-finite results.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Adaptive step size for RK4 fallback
    # Start with a reasonable step and adjust based on time remaining
    h = (t1 - t0) / 10000  # 10000 steps for high accuracy
    steps = int(np.ceil((t1 - t0) / h))
    
    for i in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        t = t0 + i * h
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        if not np.all(np.isfinite(y)):
            break
            
    return y
# EVOLVE-BLOCK-END
