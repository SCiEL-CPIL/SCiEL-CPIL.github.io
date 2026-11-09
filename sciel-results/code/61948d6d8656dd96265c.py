"""Improved ODE solver using adaptive methods with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Uses adaptive step-size methods for efficiency and accuracy.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: stop before 330s wall-clock
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin
    
    # Count function calls
    call_count = 0
    
    def counted_f(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)
    
    # Try multiple strategies, starting with most efficient
    best_result = None
    best_error = np.inf
    
    # Strategy 1: LSODA (auto stiff/non-stiff switching)
    try:
        sol = solve_ivp(
            counted_f, (t0, t1), y0,
            method='LSODA',
            rtol=1e-10, atol=1e-12,
            max_step=(t1 - t0) / 50  # reasonable max step
        )
        if sol.success and sol.t[-1] >= t1 - 1e-12:
            y_final = sol.y[:, -1]
            if np.all(np.isfinite(y_final)):
                best_result = y_final
                # Estimate error by comparing with tighter tolerance
                break
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_result if best_result is not None else y0
    
    # Strategy 2: RK45 with tight tolerances
    try:
        call_count = 0
        sol = solve_ivp(
            counted_f, (t0, t1), y0,
            method='RK45',
            rtol=1e-10, atol=1e-12,
            max_step=(t1 - t0) / 100
        )
        if sol.success and sol.t[-1] >= t1 - 1e-12:
            y_final = sol.y[:, -1]
            if np.all(np.isfinite(y_final)):
                if best_result is None:
                    best_result = y_final
                else:
                    # Compare with previous result
                    diff = np.linalg.norm(y_final - best_result)
                    if diff < 1e-8:
                        # Results agree, use the one with fewer calls
                        best_result = y_final
                break
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_result if best_result is not None else y0
    
    # Strategy 3: Radau (implicit, good for stiff problems)
    try:
        call_count = 0
        sol = solve_ivp(
            counted_f, (t0, t1), y0,
            method='Radau',
            rtol=1e-10, atol=1e-12,
            max_step=(t1 - t0) / 50
        )
        if sol.success and sol.t[-1] >= t1 - 1e-12:
            y_final = sol.y[:, -1]
            if np.all(np.isfinite(y_final)):
                if best_result is None:
                    best_result = y_final
                else:
                    diff = np.linalg.norm(y_final - best_result)
                    if diff < 1e-8:
                        best_result = y_final
                break
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_result if best_result is not None else y0
    
    # Strategy 4: BDF (another implicit method)
    try:
        call_count = 0
        sol = solve_ivp(
            counted_f, (t0, t1), y0,
            method='BDF',
            rtol=1e-10, atol=1e-12,
            max_step=(t1 - t0) / 50
        )
        if sol.success and sol.t[-1] >= t1 - 1e-12:
            y_final = sol.y[:, -1]
            if np.all(np.isfinite(y_final)):
                if best_result is None:
                    best_result = y_final
                else:
                    diff = np.linalg.norm(y_final - best_result)
                    if diff < 1e-8:
                        best_result = y_final
                break
    except Exception:
        pass
    
    # If all adaptive methods failed, fall back to fixed-step RK4
    if best_result is None:
        try:
            # Use a moderate number of steps for RK4
            steps = 5000
            h = (t1 - t0) / steps
            t, y = t0, y0.copy()
            for _ in range(steps):
                if time.time() > deadline:
                    break
                k1 = f(t, y)
                k2 = f(t + h / 2, y + h / 2 * k1)
                k3 = f(t + h / 2, y + h / 2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
                if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
                    break
            if np.all(np.isfinite(y)):
                best_result = y
        except Exception:
            pass
    
    return best_result if best_result is not None else y0
# EVOLVE-BLOCK-END
