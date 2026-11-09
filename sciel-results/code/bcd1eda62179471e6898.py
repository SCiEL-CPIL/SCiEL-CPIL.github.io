# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use a high-order explicit Runge-Kutta method (Dormand-Prince 8(5,3))
    # This is generally very efficient and accurate for non-stiff problems.
    # If the problem is stiff, this might take many steps, but for N=100, 
    # it's likely manageable. We use tight tolerances.
    
    # Try DOP853 first
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14,
            max_step=(t1-t0)/100  # Limit step size to ensure accuracy
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to Radau (implicit, good for stiff)
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14,
            max_step=(t1-t0)/100
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: RK4
    steps = 10000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
