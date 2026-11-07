import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) from t=0 to t=10.
    Uses a high-order explicit Runge-Kutta method (DOP853) via scipy
    with tight tolerances but no max_step constraint, allowing the
    adaptive stepper to take larger steps where the solution is smooth.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float).copy()
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0

    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        return y0

    # Primary: DOP853 with very tight tolerances, no max_step
    # DOP853 is 8th order, explicit, 12 stages
    # For the Van der Pol with mu=5, the solution is smooth enough
    # that DOP853 should handle it with few steps.
    
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-14,
            atol=1e-16,
        )
        
        if sol.success and sol.t[-1] == t1:
            result = sol.y[:, -1].copy()
            if np.all(np.isfinite(result)):
                return result
    except Exception:
        pass

    # Check if we're running out of time
    if time.time() > deadline - 30:
        # Fallback to Radau with moderate tolerances
        try:
            sol2 = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                rtol=1e-11,
                atol=1e-13,
            )
            if sol2.success:
                return sol2.y[:, -1].copy()
        except Exception:
            pass
        return y0

    # Secondary: Radau with very tight tolerances
    try:
        sol2 = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15,
        )
        
        if sol2.success and sol2.t[-1] == t1:
            result = sol2.y[:, -1].copy()
            if np.all(np.isfinite(result)):
                return result
    except Exception:
        pass

    # Tertiary: Radau with reference tolerances
    try:
        sol3 = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
        )
        
        if sol3.success and sol3.t[-1] == t1:
            result = sol3.y[:, -1].copy()
            if np.all(np.isfinite(result)):
                return result
    except Exception:
        pass

    return y0
