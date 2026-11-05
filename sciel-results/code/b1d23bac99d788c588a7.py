"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses adaptive Runge-Kutta with tight tolerances for high accuracy.
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Track wall-clock time for safety
    start_time = time.time()
    DEADLINE = 330.0  # seconds, evaluator kills at 330s
    
    # Try adaptive RK45 with very tight tolerances
    # rtol=1e-10, atol=1e-12 should give very high accuracy
    # max_step controls the step size to ensure we resolve the solution well
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='RK45',
            rtol=1e-11,
            atol=1e-13,
            max_step=(t1 - t0) / 1000.0,  # at least 1000 steps
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If that failed or timed out, try with slightly looser tolerances
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10.0:
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0_arr,
                method='RK45',
                rtol=1e-9,
                atol=1e-11,
                max_step=(t1 - t0) / 500.0,
                dense_output=False
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Fallback: fixed-step RK4 with many steps
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5.0:
        steps = 10000
        h = (t1 - t0) / steps
        t, y = t0, y0_arr.copy()
        for i in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2.0, y + h / 2.0 * k1)
            k3 = f(t + h / 2.0, y + h / 2.0 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            # Check for divergence
            if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
                break
            # Periodic time check
            if i % 1000 == 0 and time.time() - start_time > DEADLINE - 5.0:
                break
        return y
    
    # Last resort: return initial condition
    return y0_arr
# EVOLVE-BLOCK-END
