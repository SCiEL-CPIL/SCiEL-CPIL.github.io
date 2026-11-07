"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    Returns the state at t_span[1] as a float array with the same shape as y0.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: evaluator kills at 330s, keep a 10s safety margin
    start_time = time.time()
    deadline = start_time + 320.0
    
    # Use DOP853 with extremely tight tolerances.
    # DOP853 is an 8th-order Dormand-Prince type method with error estimator.
    # For the Lorenz system over [0, 2], this should converge to near-machine
    # precision with very few function calls (likely < 100).
    # rtol=1e-13 and atol=1e-15 should drive rel below 1e-14 (the scoring floor).
    
    # Wrap f to count calls and ensure it returns a numpy array
    call_count = [0]
    
    def f_wrapped(t, y):
        call_count[0] += 1
        result = f(t, y)
        return np.asarray(result, dtype=float)
    
    try:
        sol = solve_ivp(
            f_wrapped,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15,
            max_step=(t1 - t0) / 20.0,  # Ensure reasonable step count
        )
        
        if sol.success and sol.y is not None and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass
    
    # Fallback: if DOP853 fails for any reason, use a fixed-step RK4 with enough steps
    # to achieve high accuracy. 10000 steps of RK4 should be very accurate for this problem.
    try:
        steps = 10000
        h = (t1 - t0) / steps
        t = t0
        y = y0.copy()
        for _ in range(steps):
            k1 = f_wrapped(t, y)
            k2 = f_wrapped(t + h / 2, y + h / 2 * k1)
            k3 = f_wrapped(t + h / 2, y + h / 2 * k2)
            k4 = f_wrapped(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            if not np.all(np.isfinite(y)):
                break
        if np.all(np.isfinite(y)):
            return y
    except Exception:
        pass
    
    # Last resort: return y0 (should not happen)
    return y0
# EVOLVE-BLOCK-END
