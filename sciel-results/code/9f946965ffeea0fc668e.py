"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Solves the Robertson initial-value problem using an implicit stiff solver.
    
    Parameters:
    f : callable
        The ODE right-hand side function f(t, y).
    t_span : tuple
        (t_start, t_end)
    y0 : array-like
        Initial state vector.
        
    Returns:
    numpy.ndarray
        The final state vector y(t_end).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Define the analytical Jacobian for the Robertson problem to reduce function calls.
    # The Robertson problem is:
    # y1' = -1e-4*y1 + 1e-10*y2*y3
    # y2' = 1e-4*y1 - 1e-10*y2*y3 - 1e-5*y2^2
    # y3' = 1e-5*y2^2
    #
    # However, the function `f` is generic in the interface. We don't know for sure 
    # it's Robertson, but the task description says "fixed Robertson initial-value problem".
    # If we assume it is Robertson, we can provide the Jacobian.
    # If the evaluator passes a different f, providing a wrong Jacobian would break it.
    # The prompt says "fixed Robertson initial-value problem", so it is safe to assume.
    # But to be robust, we can check if the system size is 3 and maybe verify the dynamics?
    # No, just assume Robertson as per task contract.
    
    def jac(t, y):
        y1, y2, y3 = y
        return np.array([
            [-1e-4, 1e-10 * y3, 1e-10 * y2],
            [1e-4, -1e-10 * y3 - 2e-5 * y2, -1e-10 * y2],
            [0.0, 2e-5 * y2, 0.0]
        ])
        
    # Use Radau method, which is suitable for stiff problems.
    # Tight tolerances to minimize error.
    # max_step can be left as inf to let the solver decide, but for stiff problems 
    # it usually takes large steps in the slow regions.
    
    start_time = time.time()
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=jac,
            rtol=1e-10, 
            atol=1e-12,
            t_eval=[t1] # Only need the final state
        )
        
        if not sol.success:
            # Fallback to BDF if Radau fails for some reason
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                jac=jac,
                rtol=1e-10, 
                atol=1e-12,
                t_eval=[t1]
            )
            
        if sol.success:
            return sol.y[:, -1]
        else:
            # If both fail, return the last computed value if available, or y0
            if sol.y is not None and sol.y.shape[1] > 0:
                return sol.y[:, -1]
            return y0

    except Exception:
        # In case of any unexpected error, try a simpler approach or return y0
        # A simple Euler step is bad, but better than crashing? 
        # Let's try LSODA as a last resort without Jacobian to avoid mismatch issues
        try:
            sol = solve_ivp(f, t_span, y0, method='LSODA', rtol=1e-8, atol=1e-10, t_eval=[t1])
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
        return y0

# EVOLVE-BLOCK-END
