"""ODE solver: Robertson problem using high-accuracy implicit methods."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    
    Uses Radau (implicit, stiff-capable) with very tight tolerances for high accuracy.
    Tracks function calls via a wrapper.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    call_count = [0]
    start_time = time.time()
    deadline = 320.0  # safety margin before 330s kill
    
    def wrapped_f(t, y):
        call_count[0] += 1
        if call_count[0] > 2000000:
            raise RuntimeError("Too many function calls")
        return f(t, y)
    
    # Try Radau with very tight tolerances first
    # Robertson problem is stiff, so implicit methods are ideal
    try:
        sol = solve_ivp(
            wrapped_f,
            [t0, t1],
            y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 1000,
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: Radau with slightly looser tolerances
    try:
        sol = solve_ivp(
            wrapped_f,
            [t0, t1],
            y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 500,
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: BDF method
    try:
        sol = solve_ivp(
            wrapped_f,
            [t0, t1],
            y0,
            method='BDF',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 1000,
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: LSODA
    try:
        sol = solve_ivp(
            wrapped_f,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=1e-12,
            atol=1e-14,
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If all scipy methods fail, use a simple RK4 as last resort
    h = (t1 - t0) / 10000
    t, y = t0, y0.copy()
    steps = int(round((t1 - t0) / h))
    for _ in range(steps):
        if time.time() > deadline:
            break
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
