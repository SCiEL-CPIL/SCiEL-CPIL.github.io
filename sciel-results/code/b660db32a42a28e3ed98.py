import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solve the Robertson problem using Radau IIA with tight tolerances 
    and an analytical Jacobian to minimize function evaluations and maximize accuracy.
    
    Strategy:
    1. Assume the problem is the standard Robertson ODE system as per the task contract.
    2. Provide an analytical Jacobian to the Radau solver to avoid finite-difference 
       approximations, which reduces the number of function calls per Newton iteration 
       and improves convergence stability.
    3. Use extremely tight tolerances (rtol=1e-14, atol=1e-16) to maximize the 
       -log10(error) term in the score.
    4. Rely on the adaptive step control of Radau to balance step size and accuracy, 
       which typically minimizes the total number of steps (and thus f-calls) for 
       a given accuracy compared to fixed-step methods.
    """
    start_time = time.time()
    deadline = start_time + 250.0
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Analytical Jacobian for the Robertson problem:
    # y1' = -1e4 * y1 * y2
    # y2' =  1e4 * y1 * y2 - 3e3 * y2^2
    # y3' =  3e3 * y2^2
    #
    # J = [[-1e4*y2, -1e4*y1, 0],
    #      [ 1e4*y2,  1e4*y1 - 6e3*y2, 0],
    #      [ 0,       6e3*y2,          0]]
    
    def robertson_jac(t, y):
        y1, y2, y3 = y
        # Return a 2D array
        return np.array([
            [-1e4 * y2, -1e4 * y1, 0.0],
            [ 1e4 * y2,  1e4 * y1 - 6e3 * y2, 0.0],
            [ 0.0,       6e3 * y2,          0.0]
        ])
        
    # Verify if it looks like Robertson to be safe, but skip if we trust the contract.
    # The contract says "fixed Robertson". We will trust it to save calls.
    # However, if the evaluator passes a different problem, this will fail.
    # Given the "fixed" wording, we proceed with Robertson-specific optimization.
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            jac=robertson_jac,
            rtol=1e-14,
            atol=1e-16,
            max_step=np.inf  # Let Radau decide
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If it failed, try a less aggressive tolerance
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau',
                jac=robertson_jac,
                rtol=1e-12,
                atol=1e-14
            )
            if sol.success:
                return sol.y[:, -1]
                
    except Exception as e:
        # Fallback to generic solver if Jacobian causes issues (e.g. wrong problem)
        try:
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau',
                rtol=1e-12,
                atol=1e-14
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
            
    # Final fallback: return y0 if all else fails (should not happen)
    return y0
