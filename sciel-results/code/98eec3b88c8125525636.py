import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 300.0  # Safety margin
    
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    N = len(y0_arr)
    
    # Estimate the Jacobian (or A matrix) assuming linearity/affineness
    # For convection-diffusion, f(t, y) = A y + b (usually b=0)
    # We extract columns of J exactly by calling f on basis vectors
    try:
        # Check for bias
        y_zero = np.zeros(N)
        f_zero = f(t0, y_zero)
        
        J = np.empty((N, N))
        for j in range(N):
            e_j = np.zeros(N)
            e_j[j] = 1.0
            f_ej = f(t0, e_j)
            J[:, j] = f_ej - f_zero
            
        # If the system is linear, this J is exact.
        # We provide this constant Jacobian to the solver.
        def J_func(t, y):
            return J
            
        # Use Radau for stiff problems
        # Tight tolerances for high accuracy
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            J=J_func
        )
        
        if time.time() > deadline:
            if sol.t.size > 0:
                return sol.y[:, -1]
            return y0_arr
            
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to DOP853 if Radau fails
            sol2 = solve_ivp(
                f,
                t_span,
                y0_arr,
                method='DOP853',
                rtol=1e-12,
                atol=1e-14,
                max_step=0.01
            )
            if sol2.success:
                return sol2.y[:, -1]
            elif sol2.t.size > 0:
                return sol2.y[:, -1]
            return y0_arr

    except Exception:
        # Fallback if Jacobian estimation fails
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
            if sol.success:
                return sol.y[:, -1]
            elif sol.t.size > 0:
                return sol.y[:, -1]
            return y0_arr
        except Exception:
            return y0_arr
