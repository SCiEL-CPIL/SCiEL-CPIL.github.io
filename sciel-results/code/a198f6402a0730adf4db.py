"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We'll use scipy's high-order DOP853 solver with tight tolerances
# to minimize both relative error and function call count.
# The reference uses DOP853 with rtol=1e-11, atol=1e-13.
# Using the same solver with the same tolerances should give near-zero
# relative error and a reasonable call count.

def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system from t_span[0] to t_span[1]
    with initial condition y0, returning the state at t_span[1].
    
    Uses scipy.integrate.solve_ivp with DOP853 and tight tolerances
    to minimize both error and function evaluations.
    """
    from scipy.integrate import solve_ivp
    
    t_start = time.time()
    deadline = 330.0  # seconds
    
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use DOP853 with very tight tolerances to match the reference
    # rtol=1e-11, atol=1e-13 as specified in the task
    result = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
        max_step=0.1  # reasonable max step to ensure accuracy
    )
    
    if not result.success:
        # Fallback: try with even tighter tolerances or different method
        result = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.05
        )
    
    if result.success and result.y.shape[1] > 0:
        y_end = result.y[:, -1]
    else:
        # Last resort fallback: simple RK4 with many steps
        y_end = _rk4_fallback(f, t0, t1, y0, 5000)
    
    # Ensure the result is finite
    if not np.all(np.isfinite(y_end)):
        y_end = _rk4_fallback(f, t0, t1, y0, 5000)
    
    return np.asarray(y_end, dtype=float)


def _rk4_fallback(f, t0, t1, y0, steps):
    """Simple RK4 fallback in case scipy solver fails."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
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
