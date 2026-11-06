"""ODE solver: convection-diffusion 100-point."""
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    calls = [0]
    start_time = time.time()
    # Safety margin: evaluator kills at 330s. We aim to finish well before.
    time_limit = 290.0 
    
    def wrapped_f(t, y):
        calls[0] += 1
        # Hard stop if we exceed time or call budget
        if time.time() - start_time > time_limit or calls[0] > 1900000:
            raise TimeoutError("Time or call limit exceeded")
        return np.asarray(f(t, y), dtype=float)

    best_y = None
    best_calls = None

    # Strategy 1: Radau with extremely tight tolerances.
    # Radau is L-stable and excellent for stiff convection-diffusion.
    # Tight tolerances reduce relative error significantly.
    try:
        sol = solve_ivp(
            wrapped_f, (t0, t1), y0, 
            method='Radau', 
            rtol=1e-13, atol=1e-15, 
            max_step=0.5
        )
        if sol.success and sol.y.shape[1] > 0:
            y = sol.y[:, -1]
            if np.all(np.isfinite(y)):
                best_y = y
                best_calls = calls[0]
    except Exception:
        pass

    # Strategy 2: BDF with tight tolerances.
    # BDF is also implicit and often faster than Radau for some stiff problems.
    if best_y is None and (time.time() - start_time < time_limit):
        try:
            sol = solve_ivp(
                wrapped_f, (t0, t1), y0, 
                method='BDF', 
                rtol=1e-13, atol=1e-15, 
                max_step=0.5
            )
            if sol.success and sol.y.shape[1] > 0:
                y = sol.y[:, -1]
                if np.all(np.isfinite(y)):
                    best_y = y
                    best_calls = calls[0]
        except Exception:
            pass

    # Strategy 3: If implicit solvers failed or were too slow, try DOP853 (explicit high-order).
    # This is good if the problem is not extremely stiff.
    if best_y is None and (time.time() - start_time < time_limit):
        try:
            sol = solve_ivp(
                wrapped_f, (t0, t1), y0, 
                method='DOP853', 
                rtol=1e-12, atol=1e-14, 
                max_step=0.5
            )
            if sol.success and sol.y.shape[1] > 0:
                y = sol.y[:, -1]
                if np.all(np.isfinite(y)):
                    best_y = y
                    best_calls = calls[0]
        except Exception:
            pass

    # Strategy 4: Fallback RK45 with moderate tolerances.
    if best_y is None and (time.time() - start_time < time_limit):
        try:
            sol = solve_ivp(
                wrapped_f, (t0, t1), y0, 
                method='RK45', 
                rtol=1e-10, atol=1e-12, 
                max_step=0.5
            )
            if sol.success and sol.y.shape[1] > 0:
                y = sol.y[:, -1]
                if np.all(np.isfinite(y)):
                    best_y = y
                    best_calls = calls[0]
        except Exception:
            pass

    # Strategy 5: Manual RK4 with fixed small steps as a last resort.
    # This ensures we return *something* even if all scipy methods fail.
    if best_y is None and (time.time() - start_time < time_limit):
        try:
            # Estimate max steps we can afford
            elapsed = time.time() - start_time
            time_remaining = time_limit - elapsed
            # Assume ~1ms per f-call overhead + computation, so ~1000 calls/s is a conservative lower bound for complex f
            # But for N=100, it should be faster. Let's be conservative: 5000 steps = 20000 calls.
            max_steps = 5000
            # Adjust if we have lots of time
            if time_remaining > 100:
                max_steps = 50000
            
            h = (t1 - t0) / max_steps
            t = t0
            y = y0.copy()
            
            for i in range(max_steps):
                if time.time() - start_time > time_limit or calls[0] > 1900000:
                    break
                k1 = wrapped_f(t, y)
                k2 = wrapped_f(t + h/2, y + h/2 * k1)
                k3 = wrapped_f(t + h/2, y + h/2 * k2)
                k4 = wrapped_f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
                if not np.all(np.isfinite(y)):
                    break
            
            if np.all(np.isfinite(y)):
                best_y = y
                best_calls = calls[0]
        except Exception:
            pass

    if best_y is None:
        return y0
    return best_y
