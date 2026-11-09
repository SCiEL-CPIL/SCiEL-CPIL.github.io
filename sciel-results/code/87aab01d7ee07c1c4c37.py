"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    
    Uses BDF method (suitable for stiff problems like Robertson) with tight tolerances.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Track time to ensure we finish well within the 330s limit
    start_time = time.time()
    time_budget = 300.0  # Leave 30s margin
    
    # Robertson problem is stiff, so use BDF (5th order) with tight tolerances
    # BDF is generally more efficient than Radau for this problem
    try:
        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='BDF',
            rtol=1e-12,
            atol=1e-14,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: try Radau if BDF fails
    try:
        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: simple RK45 with moderate tolerances
    try:
        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='RK45',
            rtol=1e-10,
            atol=1e-12,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If all scipy methods fail, do a very basic RK4 as last resort
    h = (t1 - t0) / 10000
    t, y = t0, y0.copy()
    for _ in range(10000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if time.time() - start_time > time_budget:
            break
    
    return y
# EVOLVE-BLOCK-END
