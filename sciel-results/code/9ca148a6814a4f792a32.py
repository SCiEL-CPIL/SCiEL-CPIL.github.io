"""ODE integrator for the harmonic oscillator using high-order DOP853."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses DOP853 method with high precision to match the reference solution closely.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Track time to ensure we finish well within limits
    start_time = time.time()
    
    # Use DOP853 with tolerances matching the reference generator
    # This minimizes relative error against the reference while keeping call count low
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13
    )
    
    # Check for errors
    if not sol.success:
        # Fallback to a robust, though potentially less optimal, method if DOP853 fails
        # This is highly unlikely for a harmonic oscillator, but good for safety
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='RK45',
            rtol=1e-10,
            atol=1e-12
        )
        if not sol.success:
            raise RuntimeError("Integration failed")
            
    # Return the state at the end of the time span
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
