"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrates the Van der Pol oscillator using an adaptive implicit Runge-Kutta method (Radau).
    This method is highly efficient for stiff problems and achieves high accuracy with fewer
    function calls compared to fixed-step explicit methods.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Start time for deadline guard
    start_time = time.time()
    deadline = 300.0 # 300 seconds limit, keep a small margin
    
    # Define tolerances to match the reference solution's precision
    # rtol=1e-11 and atol=1e-13 are specified in the task description for y_ref
    rtol = 1e-11
    atol = 1e-13
    
    # Use Radau method which is an implicit Runge-Kutta method of order 5
    # It is well-suited for stiff problems like Van der Pol with mu=5
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=rtol, 
            atol=atol,
            max_step=1.0 # Prevent extremely large steps if needed, though Radau adapts well
        )
        
        if sol.success:
            # Return the state at the end of the interval
            y_end = sol.y[:, -1]
            # Ensure it's a finite vector
            if np.all(np.isfinite(y_end)):
                return y_end
        else:
            # If Radau fails, we need a fallback. 
            # However, for this specific IVP, Radau is very robust.
            # A simple fallback: retry with slightly relaxed tolerances or different method?
            # Let's try BDF as another implicit method, or just return the last computed point if it exists.
            if sol.t.size > 0 and sol.y.shape[1] > 0:
                y_end = sol.y[:, -1]
                if np.all(np.isfinite(y_end)):
                    return y_end
            
            # If all else fails, try a fixed step RK4 with high resolution as a last resort
            # This is unlikely to be needed but ensures we don't crash.
            steps = 10000
            h = (t1 - t0) / steps
            t, y = t0, y0.copy()
            for _ in range(steps):
                if time.time() > deadline - 5:
                    return y # Return best effort
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y

    except Exception:
        # In case of any unexpected error, fallback to RK4
        steps = 10000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            if time.time() > deadline - 5:
                return y
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y

# EVOLVE-BLOCK-END
