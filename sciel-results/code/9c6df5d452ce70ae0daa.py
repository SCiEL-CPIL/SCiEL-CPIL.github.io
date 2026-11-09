import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    h_total = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    start_time = time.time()
    deadline = start_time + 290.0 # 10s margin
    
    # 1. Probe the system to see if it's a constant-coefficient linear system
    # y' = A y + b
    A = None
    b = None
    
    try:
        # Check if system is homogeneous by calling f(t0, 0)
        # If it's close to zero, assume b=0 to save a call
        f_zero = f(t0, np.zeros(N))
        
        # Heuristic: if norm is very small, assume homogeneous
        if np.linalg.norm(f_zero) < 1e-12 * (1 + np.linalg.norm(y0)):
            b = np.zeros(N)
            # We still need to find A.
            # But wait, if b=0, then f(t, y) = A y.
            # We can find A by calling f(t, e_j).
            A = np.zeros((N, N))
            for j in range(N):
                e_j = np.zeros(N)
                e_j[j] = 1.0
                val = f(t0, e_j)
                A[:, j] = val
        else:
            # Non-homogeneous case
            b = f_zero
            A = np.zeros((N, N))
            for j in range(N):
                e_j = np.zeros(N)
                e_j[j] = 1.0
                val = f(t0, e_j)
                A[:, j] = val - b
            
        # Verify the model with a single test point to ensure linearity and autonomy
        # Using y0 itself is a good test
        y_test = y0.copy()
        f_test = f(t1, y_test)
        predicted = A @ y_test + b
        error_check = np.linalg.norm(f_test - predicted) / (1 + np.linalg.norm(predicted))
        
        # If the model holds, use exact solution
        if error_check < 1e-8:
            # Construct augmented system for exact solution
            # z' = [ A  b ]
            #     [ 0  0 ]
            # z = [y]
            #     [1]
            A_aug = np.zeros((N + 1, N + 1))
            A_aug[:N, :N] = A
            A_aug[:N, N] = b
            # Last row is 0
            
            z0 = np.zeros(N + 1)
            z0[:N] = y0
            z0[N] = 1.0
            
            try:
                # Compute matrix exponential
                exp_Ah = la.expm(A_aug * h_total)
                z_end = exp_Ah @ z0
                y_end = z_end[:N]
                
                if np.all(np.isfinite(y_end)):
                    return y_end
            except Exception:
                pass # Fall through to time stepping
                
    except Exception:
        # If probing fails completely, A and b might be None or partial
        A = None
        b = None

    # 2. Fallback: Use scipy.integrate.solve_ivp with Radau for stiff systems
    # If the system is not LTI, the static Jacobian A from t0 may be inaccurate.
    # We will let solve_ivp estimate the Jacobian (finite difference) to adapt to time-varying systems.
    # This is more accurate for non-autonomous systems, though potentially more expensive.
    
    try:
        # Radau is good for stiff problems
        sol = integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-10,
            t_eval=[t1]
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # If Radau fails, try BDF
            sol = integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-10, 
                atol=1e-10,
                t_eval=[t1]
            )
            if sol.success:
                return sol.y[:, -1]
    except Exception:
        pass
    
    # 3. Last resort fallback: Simple Backward Euler (kept from original but simplified)
    # This should rarely be reached
    steps = 100
    h = h_total / steps
    t = t0
    y = y0.copy()
    
    for i in range(steps):
        t_new = t + h
        y_new = y.copy()
        
        for _newton in range(10):
            f_val = f(t_new, y_new)
            J = np.zeros((N, N))
            for j in range(N):
                e_j = np.zeros(N)
                step_size = 1e-7 * max(1.0, abs(y_new[j]))
                e_j[j] = step_size
                J[:, j] = (f(t_new, y_new + e_j) - f_val) / step_size
            
            F_val = y_new - h * f_val - y
            M = np.eye(N) - h * J
            
            try:
                delta = la.solve(M, -F_val)
            except la.LinAlgError:
                delta = la.lstsq(M, -F_val)[0]
                
            y_new = y_new + delta
            
            if np.linalg.norm(delta) < 1e-10 * (1 + np.linalg.norm(y_new)):
                break
        
        y = y_new
        t = t_new
        
        if time.time() > deadline:
            break
            
    return y
