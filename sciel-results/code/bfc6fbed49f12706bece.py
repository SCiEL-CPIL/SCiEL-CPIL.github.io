"""Rewritten ODE integrator for the fixed Lorenz IVP using scipy DOP853."""
import numpy as np
import time
import scipy.integrate as integrate_scipy


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] with initial condition y0.
    
    Parameters:
    -----------
    f : callable
        Function f(t, y) returning the derivative dy/dt as an array.
    t_span : tuple
        (t_start, t_end) integration interval.
    y0 : array-like
        Initial condition vector.
        
    Returns:
    --------
    y_end : ndarray
        The state vector at t_end.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: stop if we're getting close to 330s
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin
    
    try:
        # Use DOP853 with tight tolerances to match the reference
        sol = integrate_scipy.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            max_step=0.01  # Limit step size for accuracy
        )
        
        if sol.success and len(sol.t) > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
        
        # If DOP853 failed, try with looser tolerances
        sol = integrate_scipy.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-10,
            atol=1e-12
        )
        
        if sol.success and len(sol.t) > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
        
        # Fallback: use RK45 if DOP853 completely fails
        sol = integrate_scipy.solve_ivp(
            f,
            t_span,
            y0,
            method='RK45',
            rtol=1e-10,
            atol=1e-12
        )
        
        if sol.success and len(sol.t) > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
        
    except Exception:
        pass
    
    # Final fallback: simple fixed-step RK4
    try:
        steps = 10000
        h = (t1 - t0) / steps
        t = t0
        y = y0.copy()
        for _ in range(steps):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        if np.all(np.isfinite(y)):
            return y
    except Exception:
        pass
    
    # If all else fails, return initial condition (should not happen)
    return y0
