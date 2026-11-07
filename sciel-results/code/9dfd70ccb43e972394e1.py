"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using SciPy's DOP853.
    """
    # Use the same method and tolerances as the reference to ensure high accuracy
    # DOP853 is an 8th/7th order method, very efficient.
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13
    )
    
    if not sol.success:
        # Fallback if solve_ivp fails for some reason, though unlikely
        # Use a simple fixed step RK4 with many steps as a backup
        t0, t1 = t_span
        h = (t1 - t0) / 10000
        t = t0
        y = np.array(y0, dtype=float)
        for _ in range(10000):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
        
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
