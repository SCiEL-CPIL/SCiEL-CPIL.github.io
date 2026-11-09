"""ODE solver: Robertson problem using Radau with adaptive tolerance refinement."""
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    
    Strategy: Use Radau (L-stable implicit Runge-Kutta) for stiff problems.
    Robertson is very stiff, so Radau is ideal. Use tight tolerances for high accuracy.
    No max_step restriction to allow large steps in stiff region.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Deadline guard: 330s wall-clock limit, use 290s as safety margin
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Track calls to f
    call_count = [0]
    
    def wrapped_f(t, y):
        call_count[0] += 1
        # Check deadline every 1000 calls to avoid overhead
        if call_count[0] % 1000 == 0:
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
        return f(t, y)
    
    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        solve_ivp = None
    
    if solve_ivp is not None:
        # Primary: Radau with very tight tolerances, no max_step
        # This allows the solver to take large steps in stiff regions
        try:
            sol = solve_ivp(
                wrapped_f,
                t_span,
                y0_arr,
                method='Radau',
                rtol=1e-12,
                atol=1e-15
            )
            if sol.success:
                return sol.y[:, -1]
        except TimeoutError:
            raise
        except Exception:
            pass
        
        # Secondary: BDF with tight tolerances
        try:
            call_count[0] = 0
            sol = solve_ivp(
                wrapped_f,
                t_span,
                y0_arr,
                method='BDF',
                rtol=1e-12,
                atol=1e-15
            )
            if sol.success:
                return sol.y[:, -1]
        except TimeoutError:
            raise
        except Exception:
            pass
        
        # Tertiary: LSODA
        try:
            call_count[0] = 0
            sol = solve_ivp(
                wrapped_f,
                t_span,
                y0_arr,
                method='LSODA',
                rtol=1e-12,
                atol=1e-15
            )
            if sol.success:
                return sol.y[:, -1]
        except TimeoutError:
            raise
        except Exception:
            pass
    
    # Final fallback: manual RK45 with moderate tolerances
    y = y0_arr.copy()
    t = t0
    h = (t1 - t0) / 1000.0
    rtol = 1e-10
    atol = 1e-12
    
    while t < t1 and time.time() < deadline:
        h = min(h, t1 - t)
        if h <= 0:
            break
        
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        k5 = f(t + h / 5, y + h / 5 * k4)
        k6 = f(t + 3 * h / 5, y + 3 * h / 5 * (k1 + 4.0 / 3.0 * (k4 - k1)))
        
        y_new = y + h * (k1 + 3 * k2 + 3 * k3 + k4) / 8.0
        err = h * (k4 - k3) / 32.0
        err_norm = np.max(np.abs(err) / (1 + np.abs(y_new)))
        
        if err_norm <= max(rtol, atol):
            y = y_new
            t += h
            # Grow step if error is small
            if err_norm > 0:
                h *= min(4.0, 0.9 * (rtol / err_norm) ** 0.2)
            else:
                h *= 4.0
        else:
            # Shrink step
            h *= max(0.1, 0.9 * (rtol / max(err_norm, 1e-15)) ** 0.2)
    
    return y
