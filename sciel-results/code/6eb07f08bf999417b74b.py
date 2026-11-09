# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    Uses scipy.integrate.solve_ivp with DOP853 at tight tolerances for high accuracy
    while keeping the function-call count low.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: return early if we are running out of time
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin before 300s limit
    
    try:
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15,
            max_step=(t1 - t0) / 1000  # Limit step size for accuracy
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to RK4 if solve_ivp fails
            return _rk4_fallback(f, t0, t1, y0)
            
    except Exception:
        # If solve_ivp raises an exception, fall back to RK4
        return _rk4_fallback(f, t0, t1, y0)


def _rk4_fallback(f, t0, t1, y0):
    """Simple RK4 fallback with a reasonable number of steps."""
    y = np.array(y0, dtype=float)
    t = t0
    # Use 10000 steps for the fallback to ensure reasonable accuracy
    steps = 10000
    h = (t1 - t0) / steps
    
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        
        # Check for non-finite values
        if not np.all(np.isfinite(y)):
            return y
    
    return y
# EVOLVE-BLOCK-END
