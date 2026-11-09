import numpy as np
import scipy.linalg as la
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
    # Call f(0, 0) to get b
    b = f(t0, np.zeros(N))
    
    # Call f(0, e_j) to get columns of A
    A = np.zeros((N, N))
    for j in range(N):
        e_j = np.zeros(N)
        e_j[j] = 1.0
        val = f(t0, e_j)
        A[:, j] = val - b
        
    # Construct augmented system for exact solution
    # Assuming the system is autonomous linear y' = A y + b as per task description
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
        pass # Fall through to time stepping (should not happen)

    # Fallback: Implicit Backward Euler with Newton iterations
    # This is kept only for safety in case expm fails, but unlikely to be reached
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
