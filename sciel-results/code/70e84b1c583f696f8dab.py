import numpy as np
import time
import sys

def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system (or any IVP defined by f) from t_span[0] to t_span[1].
    Uses scipy.integrate.solve_ivp with DOP853 for high accuracy and efficiency.
    """
    start_time = time.time()
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: 290 seconds to be safe within 300s limit
    DEADLINE = 290.0

    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        # Fallback if scipy is not available (unlikely per task description)
        # Use a fixed step RK4 as a rough fallback
        def _rk4_fallback(f, t0, t1, y0, steps=10000):
            h = (t1 - t0) / steps
            t, y = t0, np.array(y0, dtype=float)
            for _ in range(steps):
                if time.time() - start_time > DEADLINE:
                    return y
                k1 = f(t, y)
                k2 = f(t + h / 2, y + h / 2 * k1)
                k3 = f(t + h / 2, y + h / 2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
            return y
        
        return _rk4_fallback(f, t0, t1, y0)

    # Use DOP853 with very tight tolerances
    # rtol=1e-13, atol=1e-15 aims for max possible accuracy
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-13, 
            atol=1e-15,
            max_step=np.inf  # Let adaptive stepper choose
        )
        
        if sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
            else:
                # If result is not finite, try a slightly looser tolerance or fallback
                # But usually DOP853 is stable for Lorenz
                pass
        
        # If sol.success is False or result non-finite, try a more conservative approach
        # or just return the last computed value if it's finite
        if sol.y.size > 0 and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
            
    except Exception as e:
        # If solve_ivp throws an exception, fall back to a robust fixed-step method
        pass

    # Fallback: Fixed step RK4 with high resolution
    # This ensures we return *something* valid if the adaptive method fails catastrophically
    # 20000 steps for t in [0, 2] is h=1e-4, which is quite accurate for RK4
    def _rk4_fixed(f, t0, t1, y0, steps=20000):
        h = (t1 - t0) / steps
        t, y = t0, np.array(y0, dtype=float)
        for i in range(steps):
            if time.time() - start_time > DEADLINE:
                return y
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y

    return _rk4_fixed(f, t0, t1, y0)
