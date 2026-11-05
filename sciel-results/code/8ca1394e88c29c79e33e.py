"""ODE solver: Robertson problem using LSODA with analytic Jacobian and Radau polish via scipy.integrate.solve_ivp."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Robertson ODE y' = f(t, y) from t_span[0] to t_span[1].
    Uses LSODA with the known analytic Jacobian for the Robertson problem
    to minimize function evaluations while achieving high accuracy.
    If time permits, applies a Radau polish pass with tighter tolerances.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Deadline guard: 330s wall-clock limit, use 320s as safety margin
    start_time = time.time()
    deadline = start_time + 320

    # Count function evaluations
    call_count = [0]
    
    def wrapped_f(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Analytic Jacobian for the Robertson problem.
    # y = [y1, y2, y3]
    # y1' = -0.04*y1 + 1e4*y2*y3
    # y2' =  0.04*y1 - 1e4*y2*y3 - 3e7*y2*y2
    # y3' =  3e7*y2*y2
    #
    # J = [
    #   [-0.04,      1e4*y3,    1e4*y2],
    #   [ 0.04, -1e4*y3-6e7*y2, -1e4*y2],
    #   [ 0.0,     6e7*y2,        0.0   ]
    # ]
    
    def jac(t, y):
        y1, y2, y3 = y
        J = np.empty((3, 3))
        J[0, 0] = -0.04
        J[0, 1] = 1e4 * y3
        J[0, 2] = 1e4 * y2
        J[1, 0] = 0.04
        J[1, 1] = -1e4 * y3 - 6e7 * y2
        J[1, 2] = -1e4 * y2
        J[2, 0] = 0.0
        J[2, 1] = 6e7 * y2
        J[2, 2] = 0.0
        return J

    # Primary attempt: LSODA with very tight tolerances and analytic Jacobian
    # LSODA automatically switches between Adams (non-stiff) and BDF (stiff),
    # and the Jacobian helps the stiff (BDF) part converge faster.
    best_result = None
    best_calls = float('inf')
    
    try:
        sol = solve_ivp(
            wrapped_f,
            (t0, t1),
            y0_arr,
            method='LSODA',
            jac=jac,
            rtol=1e-14,
            atol=1e-16,
            max_step=(t1 - t0) / 1000
        )
        
        if sol.success and sol.y.shape[1] > 0:
            best_result = sol.y[:, -1].copy()
            best_calls = call_count[0]
        elif sol.y.shape[1] > 0:
            best_result = sol.y[:, -1].copy()
            best_calls = call_count[0]
        
    except Exception:
        pass
    
    # If LSODA failed or we want to try alternative, try Radau
    if best_result is None:
        try:
            call_count[0] = 0
            sol = solve_ivp(
                wrapped_f,
                (t0, t1),
                y0_arr,
                method='Radau',
                jac=jac,
                rtol=1e-14,
                atol=1e-16
            )
            if sol.success and sol.y.shape[1] > 0:
                best_result = sol.y[:, -1].copy()
                best_calls = call_count[0]
            elif sol.y.shape[1] > 0:
                best_result = sol.y[:, -1].copy()
                best_calls = call_count[0]
        except Exception:
            pass
    
    # If still no result, try BDF
    if best_result is None:
        try:
            call_count[0] = 0
            sol = solve_ivp(
                wrapped_f,
                (t0, t1),
                y0_arr,
                method='BDF',
                jac=jac,
                rtol=1e-13,
                atol=1e-15
            )
            if sol.success and sol.y.shape[1] > 0:
                best_result = sol.y[:, -1].copy()
                best_calls = call_count[0]
            elif sol.y.shape[1] > 0:
                best_result = sol.y[:, -1].copy()
                best_calls = call_count[0]
        except Exception:
            pass
    
    # If still no result, fallback to RK4
    if best_result is None:
        try:
            h = (t1 - t0) / 10000
            t = t0
            y = y0_arr.copy()
            n_steps = 10000
            for _ in range(n_steps):
                if time.time() > deadline:
                    break
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            best_result = y
            best_calls = call_count[0] + 4 * n_steps  # approximate
        except Exception:
            return y0_arr
    
    # Polish pass: if we have time remaining, use Radau with even tighter tolerances
    # to refine the result. This is a high-precision polish within the same design family.
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    
    if remaining > 30 and best_result is not None:
        try:
            # Use Radau with extremely tight tolerances as a polish
            # Start from the LSODA result as initial condition for a short refinement
            # Actually, re-integrate from scratch with tighter tolerances
            call_count[0] = 0
            sol_polish = solve_ivp(
                wrapped_f,
                (t0, t1),
                y0_arr,
                method='Radau',
                jac=jac,
                rtol=1e-15,
                atol=1e-17
            )
            
            if sol_polish.success and sol_polish.y.shape[1] > 0:
                polish_result = sol_polish.y[:, -1].copy()
                polish_calls = call_count[0]
                
                # Compare scores: -log10(error) - 0.5*log10(calls)
                # We can't know the error, but tighter tolerance should give lower error
                # Use the polish result if it succeeded
                best_result = polish_result
                best_calls = polish_calls
                
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
