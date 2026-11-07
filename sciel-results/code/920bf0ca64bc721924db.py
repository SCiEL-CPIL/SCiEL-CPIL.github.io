import numpy as np
import scipy.integrate
import time


def integrate(f, t_span, y0):
    start_time = time.time()
    
    # Use DOP853 with tight tolerances to match the reference solution closely
    # while keeping the number of function calls low.
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13
        )
        
        # Check if we are approaching the time limit
        elapsed = time.time() - start_time
        if elapsed < 300:
            if sol.success:
                y_end = sol.y[:, -1]
                if np.all(np.isfinite(y_end)):
                    return y_end
    except Exception:
        pass
    
    # Fallback to RK4 if DOP853 fails or times out
    t0, t1 = t_span
    h = (t1 - t0) / 1000
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(1000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            break
        # Safety check
        if time.time() - start_time > 290:
            break
    
    return y
# EVOLVE-BLOCK-END
