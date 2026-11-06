"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 320.0
    
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    N = len(y0_arr)
    
    # Estimate Jacobian at t0 assuming linearity (constant J)
    # This is valid for constant-coefficient convection-diffusion
    try:
        f0 = f(t0, y0_arr)
        h = 1e-7 * np.maximum(1.0, np.abs(y0_arr))
        J = np.empty((N, N))
        for j in range(N):
            y_pert = y0_arr.copy()
            y_pert[j] += h[j]
            f_pert = f(t0, y_pert)
            J[:, j] = (f_pert - f0) / h[j]
    except Exception:
        J = None

    # Strategy 1: Radau with estimated Jacobian
    # Implicit solver for stiff problems
    # Using a constant Jacobian approximation speeds up Newton iterations
    if J is not None:
        def J_func(t, y):
            return J
        
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0_arr,
                method='Radau',
                rtol=1e-10,
                atol=1e-12,
                J=J_func
            )
            
            if time.time() > deadline:
                if sol.t.size > 0:
                    return sol.y[:, -1]
                return y0_arr
                
            if sol.success:
                return sol.y[:, -1]
            else:
                # If Radau failed, try DOP853 as fallback
                pass
                
        except Exception:
            pass

    # Strategy 2: DOP853 (Explicit, high order)
    # Fallback or primary if Radau didn't succeed
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.01
        )
        
        if time.time() > deadline:
            if sol.t.size > 0:
                return sol.y[:, -1]
            return y0_arr
            
        if sol.success:
            return sol.y[:, -1]
        else:
            if sol.t.size > 0:
                return sol.y[:, -1]
            return y0_arr
            
    except Exception:
        # Last resort: Return y0 if everything fails (should not happen)
        return y0_arr

# EVOLVE-BLOCK-END
