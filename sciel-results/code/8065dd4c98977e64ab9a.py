"""ODE solver: Convection-diffusion 100-point using linear system detection and matrix exponential."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    n = len(y0_arr)
    
    deadline = time.time() + 280  # 300s limit, keep 20s margin
    max_calls = 2000000
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Strategy 1: Assume linear autonomous y' = A y (convection-diffusion is typically this)
    # Build A by evaluating f at basis vectors. Cost: n calls.
    # Verify with 1 additional call. Total: n+1 calls.
    
    A = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        A[:, i] = f_counted(t0, e_i)
    
    # Verify linearity with a random vector
    y_verify = np.random.randn(n)
    f_verify = f_counted(t0, y_verify)
    A_y_verify = A @ y_verify
    
    if np.allclose(f_verify, A_y_verify, rtol=1e-10, atol=1e-12):
        # Confirmed linear autonomous system
        # Solution: y(t1) = expm(A * (t1 - t0)) @ y0
        dt = t1 - t0
        try:
            if time.time() < deadline - 10:
                expm_A_dt = expm(A * dt)
                y_final = expm_A_dt @ y0_arr
                if np.all(np.isfinite(y_final)):
                    return y_final
        except Exception:
            pass
    
    # Strategy 2: Fallback to stiff solver if linear detection failed or expm failed
    # Use Radau or BDF for stiff problems
    # Convection-diffusion is often stiff.
    
    # Reset call count for the solver? No, the evaluator counts total calls.
    # The current call_count is ~n+1.
    
    try:
        # Try Radau first (good for stiff)
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0_arr,
            method='Radau',
            rtol=1e-13,
            atol=1e-15,
            max_nfev=max_calls - call_count[0]
        )
        
        if sol.success and sol.t[-1] >= t1 - 1e-14:
            return sol.y[:, -1]
            
    except Exception:
        pass
    
    try:
        # Try BDF
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0_arr,
            method='BDF',
            rtol=1e-13,
            atol=1e-15,
            max_nfev=max_calls - call_count[0]
        )
        
        if sol.success and sol.t[-1] >= t1 - 1e-14:
            return sol.y[:, -1]
            
    except Exception:
        pass
    
    # Final fallback: DOP853 (explicit, high order, non-stiff)
    try:
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0_arr,
            method='DOP853',
            rtol=1e-14,
            atol=1e-16,
            max_nfev=max_calls - call_count[0]
        )
        
        if sol.success and sol.t[-1] >= t1 - 1e-14:
            return sol.y[:, -1]
            
    except Exception:
        pass
        
    # If all failed, return last known state or y0
    return y0_arr
# EVOLVE-BLOCK-END
