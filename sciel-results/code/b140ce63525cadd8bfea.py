"""ODE integrator for the Lotka-Volterra system with high accuracy and low call count."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time


def integrate(f, t_span, y0):
    """
    Integrates the ODE system y' = f(t, y) from t_span[0] to t_span[1]
    with initial condition y0, returning the state at t_end.
    
    Uses DOP853 with tight tolerances to achieve high accuracy with minimal
    function calls, maximizing the score = -log10(rel) - 0.5*log10(calls).
    
    Parameters:
    f : callable
        Function f(t, y) that returns the derivative y'.
    t_span : tuple
        (t_start, t_end)
    y0 : array-like
        Initial state vector.
        
    Returns:
    y_end : ndarray
        The state at t_end.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: must finish well before 330s wall-clock limit
    start_time = time.time()
    deadline = start_time + 300  # 300s safety margin
    
    # Strategy: Use DOP853 with very tight tolerances to ensure rel <= 1e-14,
    # which hits the scoring floor. This minimizes the rel term in the score.
    # DOP853 is an 8th-order method with 12 stages, very efficient for smooth problems.
    
    # Try progressively tighter tolerances, starting with ones that should
    # achieve rel <= 1e-14 with minimal steps
    tolerances = [
        (1e-13, 1e-15),  # Tight - should hit the 1e-14 floor
        (1e-12, 1e-14),  # Slightly looser fallback
        (1e-11, 1e-13),  # Reference tolerances
    ]
    
    best_result = None
    best_calls = float('inf')
    
    for rtol, atol in tolerances:
        if time.time() > deadline:
            break
            
        try:
            sol = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='DOP853', 
                rtol=rtol, 
                atol=atol,
                max_step=np.inf  # Allow large steps where smooth
            )
            
            if sol.success and len(sol.t) > 0:
                y_end = sol.y[:, -1]
                if np.all(np.isfinite(y_end)):
                    # Count function calls from the solver
                    # solve_ivp doesn't directly expose call count, but we can estimate
                    # from the number of steps and stages
                    n_steps = len(sol.t) - 1
                    # DOP853 uses 12 stages per step, but not all stages call f
                    # Actually, solve_ivp wraps f, so we need to count differently
                    # For now, store the result and we'll use a wrapper to count
                    best_result = y_end
                    break  # Use the first successful tight tolerance result
                    
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: use reference tolerances
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13,
            max_step=np.inf
        )
        if sol.success and len(sol.t) > 0:
            return sol.y[:, -1]
        else:
            raise RuntimeError("Integration failed")
    
    return best_result

# EVOLVE-BLOCK-END
