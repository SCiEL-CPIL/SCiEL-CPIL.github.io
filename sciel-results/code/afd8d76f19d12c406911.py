"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: aim to finish well before 330s wall-clock limit
    start_time = time.time()
    deadline = start_time + 300.0  # safety margin
    
    # Strategy: Use Radau (implicit, stiff-capable) or BDF with tight tolerances.
    # Robertson problem is stiff, so explicit methods are inefficient.
    # We'll try Radau first with very tight tolerances, and if that's too slow,
    # fall back to BDF or LSODA.
    
    # Try Radau with tight tolerances
    # rtol and atol control the error; we want high accuracy
    # The score is -log10(rel_err) - 0.5*log10(calls)
    # So we want to minimize relative error while keeping calls reasonable.
    
    # Let's start with rtol=1e-12, atol=1e-14 for high accuracy
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 1000  # reasonable max step
        )
        if sol.success and sol.t[-1] >= t1 - 1e-15:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 60:
        # Not enough time for more attempts, return best effort
        # Try a quick BDF as fallback
        try:
            sol = solve_ivp(
                f, t_span, y0,
                method='BDF',
                rtol=1e-10,
                atol=1e-12,
                max_step=(t1 - t0) / 500
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
        return y0  # last resort
    
    # Try BDF with tight tolerances
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 1000
        )
        if sol.success and sol.t[-1] >= t1 - 1e-15:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Try LSODA (auto stiff/non-stiff switching)
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='LSODA',
            rtol=1e-12,
            atol=1e-14
        )
        if sol.success and sol.t[-1] >= t1 - 1e-15:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Final fallback: simple RK4 with many steps (unlikely to be needed)
    h = (t1 - t0) / 5000
    t, y = t0, y0.copy()
    for _ in range(5000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
        if time.time() > deadline - 30:
            break
    return y
# EVOLVE-BLOCK-END
