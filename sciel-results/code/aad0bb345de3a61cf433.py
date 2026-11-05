"""ODE solver: Convection-diffusion 100-point using scipy DOP853."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Uses scipy's DOP853 (8th-order Dormand-Prince) with tight tolerances.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Wall-clock deadline guard: return best result before 330s limit
    deadline = time.time() + 300.0
    
    # Use DOP853 with very tight tolerances for high accuracy
    # rtol=1e-12, atol=1e-14 should give near-machine-precision for this problem
    sol = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='DOP853',
        rtol=1e-12,
        atol=1e-14,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback: try with slightly looser tolerances if DOP853 fails
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-10,
            atol=1e-12,
            dense_output=False
        )
    
    if sol.success:
        return sol.y[:, -1]
    
    # Last resort: use RK45 with moderate tolerances
    sol = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='RK45',
        rtol=1e-10,
        atol=1e-12,
        dense_output=False
    )
    
    if sol.success:
        return sol.y[:, -1]
    
    # Absolute last resort: manual RK4 with fixed steps
    h = (t1 - t0) / 5000
    t, y = t0, y0.copy()
    n_steps = 5000
    for _ in range(n_steps):
        if time.time() > deadline - 10.0:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    
    return y
# EVOLVE-BLOCK-END
