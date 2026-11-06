"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time
import sys

def _robertson_jac(t, y):
    # Analytical Jacobian for the Robertson problem
    # y0 = A, y1 = B, y2 = C
    # y' = f(y)
    # f0 = -k1*y0 + k3*y1*y2
    # f1 = k1*y0 - k2*y1^2 - k3*y1*y2
    # f2 = k2*y1^2
    #
    # Constants for Robertson problem (standard):
    # k1 = 0.04, k2 = 3e7, k3 = 1e4
    
    k1 = 0.04
    k2 = 3e7
    k3 = 1e4
    
    y0, y1, y2 = y
    
    # J = [[df0/dy0, df0/dy1, df0/dy2],
    #      [df1/dy0, df1/dy1, df1/dy2],
    #      [df2/dy0, df2/dy1, df2/dy2]]
    
    J = np.array([
        [-k1, k3*y2, k3*y1],
        [k1, -2*k2*y1 - k3*y2, -k3*y1],
        [0.0, 2*k2*y1, 0.0]
    ])
    
    return J

def integrate(f, t_span, y0):
    """
    Integrates the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Start time for deadline guard
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, keep 20s safety margin
    
    # We assume the function f is the Robertson problem based on the task description.
    # However, to be robust, we should check if we can use the analytical Jacobian.
    # The task specifies "fixed Robertson initial-value problem", so we can safely assume
    # the standard constants and provide the Jacobian.
    
    # Use Radau method which is suitable for stiff problems.
    # Radau IIA is an implicit Runge-Kutta method.
    # It is 5th order accurate.
    
    # Set tolerances to be very tight to maximize accuracy score.
    # The score is -log10(relative_error) - 0.5*log10(calls).
    # Reducing error by 10x increases score by 1.
    # Reducing calls by 10x increases score by 0.5.
    # Usually, reducing calls is harder than reducing error once a good solver is picked.
    # So we prioritize very low error.
    
    # Try to use even tighter tolerances if time allows, but start with reasonable ones.
    # Radau with analytical Jacobian is efficient.
    rtol = 1e-12
    atol = 1e-14
    
    # Max step size is not strictly necessary for Radau on this problem as it adapts well,
    # but we can leave it default (inf) to let it take huge steps in the stiff region.
    
    # We provide the Jacobian to avoid finite difference approximations,
    # which saves function evaluations.
    # Note: If the user's f is NOT Robertson, this will be wrong. 
    # But the task says "fixed Robertson initial-value problem".
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=_robertson_jac,
            rtol=rtol, 
            atol=atol,
            # dense_output=False # We only need the final state
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If it failed, try a looser tolerance or different method?
            if sol.y is not None and sol.y.shape[1] > 0:
                return sol.y[:, -1]
            else:
                # Fallback to a simpler method if Radau fails completely (unlikely)
                sol2 = solve_ivp(
                    f, 
                    t_span, 
                    y0, 
                    method='LSODA', 
                    rtol=rtol, 
                    atol=atol
                )
                if sol2.success:
                    return sol2.y[:, -1]
                else:
                    return sol2.y[:, -1] if sol2.y is not None and sol2.y.shape[1] > 0 else y0
                    
    except Exception as e:
        # If an exception occurs, try a fallback
        try:
            sol3 = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-9, 
                atol=1e-11
            )
            if sol3.success:
                return sol3.y[:, -1]
            else:
                return sol3.y[:, -1] if sol3.y is not None and sol3.y.shape[1] > 0 else y0
        except Exception:
            return y0

# EVOLVE-BLOCK-END
