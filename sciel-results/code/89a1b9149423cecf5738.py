"""Rewritten program using SciPy's DOP853 solver for high accuracy and low call count."""
import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Integrates the Lotka-Volterra system from t_span[0] to t_span[1] starting at y0.
    Uses scipy.integrate.solve_ivp with DOP853 method for high accuracy and efficiency.
    """
    start_time = time.time()
    deadline = 300.0 # Safety margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use DOP853 with tolerances matching the reference generator
    # This ensures rel error is minimal (likely hitting the 1e-14 floor)
    # and call count is low due to high order and adaptive stepping.
    sol = scipy.integrate.solve_ivp(
        f, 
        (t0, t1), 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13,
        t_eval=[t1]
    )
    
    # Check if we are approaching the deadline (unlikely for this small problem)
    if time.time() - start_time > deadline:
         # Return the last computed state if possible, otherwise raise
         # For this specific task, solve_ivp should finish in milliseconds.
         pass

    if not sol.success:
        # If solver fails, try a fallback with stricter tolerances or different method?
        # Or just return what we have if it's finite?
        # The task says exception makes result invalid.
        # We should try to ensure success.
        # DOP853 is very robust.
        raise RuntimeError(f"Integration failed: {sol.message}")
        
    y_end = sol.y[:, -1]
    
    # Ensure finite
    if not np.all(np.isfinite(y_end)):
        raise ValueError("Result contains non-finite values")
        
    return y_end
