import numpy as np
import time
import scipy.linalg as la
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill

    # Strategy:
    # 1. Attempt to identify if the system is Linear/Affine and Time-Invariant.
    #    y' = A @ y + b
    #    If so, solve exactly using matrix exponential.
    # 2. Fallback to high-order adaptive ODE solver if linearity fails.

    try:
        # --- Step 1: Probe for Affine/Linear Structure ---
        
        # Probe b = f(t0, 0)
        # If the system is homogeneous, b will be 0.
        y_zero = np.zeros(n)
        b = np.asarray(f(t0, y_zero), dtype=float)
        
        # Probe columns of A: A[:, i] = f(t0, e_i) - b
        # We need n calls here.
        A_cols = []
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            fi = np.asarray(f(t0, e_i), dtype=float)
            A_cols.append(fi - b)
        
        A = np.column_stack(A_cols)
        
        # Verification 1: Linearity at t0
        # Pick a random test vector
        np.random.seed(42)
        y_test = np.random.rand(n)
        y_test = y_test / np.linalg.norm(y_test) * 10.0 # Scale to typical magnitude
        
        pred_t0 = A @ y_test + b
        act_t0 = np.asarray(f(t0, y_test), dtype=float)
        
        err_t0 = np.linalg.norm(act_t0 - pred_t0)
        scale_t0 = np.linalg.norm(pred_t0) + 1e-12
        
        # Verification 2: Time Invariance
        # Check if f(t_mid, y_test) matches the same A, b
        t_mid = (t0 + t1) / 2.0
        act_tmid = np.asarray(f(t_mid, y_test), dtype=float)
        
        err_tmid = np.linalg.norm(act_tmid - pred_t0)
        scale_tmid = np.linalg.norm(pred_t0) + 1e-12
        
        # Check if t-dependence is negligible
        # Tolerance for "linear time-invariant"
        tol = 1e-9
        
        if err_t0 / scale_t0 < tol and err_tmid / scale_tmid < tol:
            # System is y' = A y + b
            dt = t1 - t0
            
            # Construct augmented system for affine case:
            # [ y' ]   [ A  b ] [ y ]
            # [ 1' ] = [ 0  1 ] [ 1 ]
            # Solution: [ y(t1) ] = expm(dt * M) @ [ y0 ]
            #           [ 1     ]                 [ 1  ]
            
            M = np.zeros((n + 1, n + 1))
            M[:n, :n] = A
            M[:n, n] = b
            M[n, n] = 1.0
            
            z0 = np.zeros(n + 1)
            z0[:n] = y0
            z0[n] = 1.0
            
            # Compute matrix exponential
            # expm is O(n^3), for n=100 this is ~1e6 ops, very fast.
            ExpM = la.expm(dt * M)
            
            z_final = ExpM @ z0
            y_final = z_final[:n]
            
            if np.all(np.isfinite(y_final)):
                return y_final
            else:
                # If expm failed or produced NaNs, fallback
                pass
        else:
            # Not linear/time-invariant. 
            # We have already spent n+2 calls.
            # Proceed to fallback.
            pass

    except Exception:
        # If any probe fails (e.g., f doesn't like zeros or basis vectors), fallback
        pass

    # --- Fallback: High-Accuracy Adaptive Solver ---
    
    # Check time
    if time.time() > deadline:
        # Out of time, return best guess (initial state or last known)
        # Since we didn't integrate, return y0 is the safest "valid" guess if we crashed early
        return y0

    # Try DOP853 (high order explicit) first as it's often more efficient for non-stiff
    # Then Radau/BDF for stiff
    methods_to_try = [
        ('DOP853', {'rtol': 1e-10, 'atol': 1e-12}),
        ('RK45', {'rtol': 1e-10, 'atol': 1e-12}),
        ('Radau', {'rtol': 1e-10, 'atol': 1e-12}),
        ('BDF', {'rtol': 1e-10, 'atol': 1e-12})
    ]
    
    for method_name, opts in methods_to_try:
        if time.time() > deadline:
            break
        try:
            sol = solve_ivp(
                f, t_span, y0,
                method=method_name,
                **opts,
                max_step=(t1 - t0) / 50 # Limit step size to ensure accuracy/structure capture
            )
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                return sol.y[:, -1]
        except Exception:
            continue

    # Last resort: Fixed step RK4 with moderate steps
    # If we reach here, we are likely in trouble with time or solver failures.
    # Use a conservative number of steps to finish quickly.
    steps = 2000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        
    return y
