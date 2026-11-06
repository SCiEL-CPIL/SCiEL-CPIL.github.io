"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Robertson ODE system using an adaptive stiff solver (Radau).
    The evaluator counts calls to f. We use a wrapper to count them.
    
    Strategy:
    - Use scipy.integrate.solve_ivp with 'Radau' method (L-stable, good for stiff problems)
    - Tight tolerances (rtol=1e-12, atol=1e-14) for high accuracy
    - Track elapsed time and return best result if approaching deadline
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: 330s wall-clock limit, use 320s safety margin
    start_time = time.time()
    deadline = start_time + 320.0
    
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Try Radau with very tight tolerances
    try:
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.1,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If Radau failed, try BDF
    try:
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0,
            method='BDF',
            rtol=1e-11,
            atol=1e-13,
            max_step=0.1,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: explicit RK45 with reasonable steps
    try:
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0,
            method='RK45',
            rtol=1e-10,
            atol=1e-12,
            max_step=0.01,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: fixed-step RK4
    h = 0.001
    n_steps = int(np.ceil((t1 - t0) / h))
    t, y = t0, y0.copy()
    for _ in range(n_steps):
        if time.time() > deadline - 5.0:
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
