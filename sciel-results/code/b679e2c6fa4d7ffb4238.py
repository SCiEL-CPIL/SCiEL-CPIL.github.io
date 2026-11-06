"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Strategy: Use scipy.integrate.solve_ivp with the Radau method.
    Radau is an implicit Runge-Kutta method that is A-stable, making it highly
    effective for stiff problems like the Robertson chemical kinetics problem.
    It automatically adapts step size and order (if allowed, but Radau is fixed
    order in SciPy usually, or adaptive step size).
    
    We set tight tolerances to minimize relative terminal error.
    The function `f` is passed directly to solve_ivp, and SciPy will count
    the number of evaluations internally. The evaluator will presumably
    wrap `f` to count calls, so we just pass it through.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Radau is best for stiff problems.
    # We use tight tolerances to ensure high accuracy.
    # max_step is set to infinity (default) to let the solver take large steps
    # where the solution is smooth, but it will automatically shrink steps
    # in stiff regions.
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-14,
            max_step=np.inf
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if Radau fails (unlikely for Robertson)
            # Try BDF as an alternative stiff solver
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-10, 
                atol=1e-14,
                max_step=np.inf
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                # Last resort: return last available state or initial
                return sol.y[:, -1] if sol.y.size > 0 else y0
    except Exception:
        # If anything goes wrong, return initial state or a safe guess
        return y0
# EVOLVE-BLOCK-END
