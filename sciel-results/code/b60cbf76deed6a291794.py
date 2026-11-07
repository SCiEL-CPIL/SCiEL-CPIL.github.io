"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system using a high-order adaptive Runge-Kutta method.
    
    Parameters:
    f : callable
        Function defining the ODE system: f(t, y) -> dy/dt
    t_span : tuple
        (t_start, t_end)
    y0 : array_like
        Initial state
    
    Returns:
    y_end : ndarray
        State at t_end
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Track deadline to avoid timeout
    start_time = time.time()
    deadline = start_time + 320  # 10s safety margin
    
    # Use DOP853 for high accuracy with relatively few steps
    # Very tight tolerances to ensure rel error is minimized
    # The reference uses rtol=1e-11, atol=1e-13, so we go tighter
    sol = solve_ivp(
        f, 
        t_span, 
        y0_arr, 
        method='DOP853', 
        rtol=1e-14, 
        atol=1e-16,
        max_step=(t1 - t0) / 20  # Ensure good resolution
    )
    
    if not sol.success:
        # Fallback: try with slightly relaxed tolerances
        sol = solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='DOP853', 
            rtol=1e-13, 
            atol=1e-15
        )
        
    if not sol.success:
        # Last resort fallback
        sol = solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='RK45', 
            rtol=1e-13, 
            atol=1e-15
        )
        
    if not sol.success:
        raise RuntimeError("Integration failed")
        
    y_end = sol.y[:, -1]
    
    # Ensure finite result
    if not np.all(np.isfinite(y_end)):
        raise ValueError("Result is not finite")
        
    return y_end
# EVOLVE-BLOCK-END
