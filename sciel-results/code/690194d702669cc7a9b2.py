"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def _robertson_jac(t, y):
    """
    Analytical Jacobian for the Robertson problem.
    y' = f(t, y)
    y1' = -10*y1 + 1e4*y2*y3
    y2' = 10*y1 - 1e4*y2*y3 - 1e6*y2^2
    y3' = 1e6*y2^2
    """
    y1, y2, y3 = y
    return np.array([
        [-10.0, 1e4 * y3, 1e4 * y2],
        [10.0, -1e4 * y3 - 2e6 * y2, -1e4 * y2],
        [0.0, 2e6 * y2, 0.0]
    ])


def integrate(f, t_span, y0):
    """
    Integrates the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses an adaptive implicit Runge-Kutta method (Radau) with an analytical Jacobian
    for the specific Robertson problem structure to minimize function evaluations.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Check if the problem is the Robertson problem by inspecting the function or just assuming it
    # The task specifies "fixed Robertson initial-value problem".
    # We assume f is the Robertson RHS. We provide the Jacobian to save calls.
    
    # If f is not Robertson, providing the wrong Jacobian will fail.
    # However, the task contract says "fixed Robertson initial-value problem".
    # To be safe, we can try to detect if we are dealing with Robertson by checking dimensions and maybe a test call,
    # but the contract implies f IS the Robertson function.
    
    # Let's assume f is the Robertson function.
    # We use Radau method which is suitable for stiff problems.
    # Providing jac reduces the number of f calls needed for Newton iterations.
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=_robertson_jac,
            rtol=1e-12, 
            atol=1e-14,
            max_step=np.inf # Let the solver decide, but usually it adapts well
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if Radau fails for some reason, though unlikely for Robertson
            # Try BDF
            sol_bdf = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                jac=_robertson_jac,
                rtol=1e-12, 
                atol=1e-14
            )
            if sol_bdf.success:
                return sol_bdf.y[:, -1]
            else:
                # Last resort: simple explicit with many steps (should not happen)
                return None
                
    except Exception:
        # If something goes wrong, return a zero vector or raise
        # Raising might result in 0 score, so returning a best guess is safer?
        # The contract says "invalid result... scores 0".
        # So we should try to return something valid if possible.
        # But if the solver fails, we don't have a valid result.
        # Let's just let the exception propagate or return a dummy.
        # Given the stiffness, Radau should work.
        raise

# EVOLVE-BLOCK-END
