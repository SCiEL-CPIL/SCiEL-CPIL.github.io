import numpy as np
import time

def find_root_system(F, x0):
    tol = 1e-8
    tol_sq = tol * tol
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    def time_ok():
        return time.time() - start_time < time_limit

    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
        
    call_count = 0
    
    def F_count(x_pt):
        nonlocal call_count
        if call_count >= 490: # Hard stop to ensure we don't exceed 500
             return None
        res = F(x_pt)
        call_count += 1
        return res

    try:
        Fx = F_count(x)
        if Fx is None:
            return x
        if not np.all(np.isfinite(Fx)):
            return x
        norm_sq = float(np.dot(Fx, Fx))
        if norm_sq <= tol_sq:
            return x
    except:
        return x

    best_x = x.copy()
    best_norm_sq = norm_sq

    # Levenberg-Marquardt parameters
    lambda_lm = 1e-2
    lambda_min = 1e-12
    lambda_max = 1e12
    max_iter = 100
    
    # Helper to compute Jacobian
    def compute_jacobian(x_pt, Fx_pt):
        # Forward difference
        h = 1e-7
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        
        try:
            Fx1 = F_count(x_pt + e1)
            Fx2 = F_count(x_pt + e2)
            if Fx1 is None or Fx2 is None:
                return None
            J = np.array([(Fx1 - Fx_pt)/h, (Fx2 - Fx_pt)/h]).T
            if not np.all(np.isfinite(J)):
                return None
            return J
        except:
            return None

    # Initial Jacobian
    J = compute_jacobian(x, Fx)
    if J is None:
        return best_x

    for i in range(max_iter):
        if not time_ok() or call_count >= 480:
            break
            
        # LM Step
        # Solve (J^T J + lambda I) delta = -J^T F
        JtJ = J.T @ J
        JtF = J.T @ Fx
        A = JtJ + lambda_lm * np.eye(2)
        b = -JtF
        
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # If singular, increase lambda and retry
            lambda_lm *= 10
            if lambda_lm > lambda_max:
                break
            continue

        x_try = x + delta
        Fx_try = F_count(x_try)
        if Fx_try is None:
            break
        if not np.all(np.isfinite(Fx_try)):
            lambda_lm *= 10
            if lambda_lm > lambda_max:
                break
            continue
            
        norm_sq_try = float(np.dot(Fx_try, Fx_try))
        
        if norm_sq_try < best_norm_sq:
            best_x = x_try.copy()
            best_norm_sq = norm_sq_try
            
        if norm_sq_try <= tol_sq:
            return x_try

        # Acceptance criteria
        if norm_sq_try < norm_sq:
            # Accept step
            x = x_try
            Fx = Fx_try
            norm_sq = norm_sq_try
            # Decrease lambda
            lambda_lm = max(lambda_lm / 10.0, lambda_min)
            
            # Recompute Jacobian? 
            # For LM, it's common to reuse J for a few steps or recompute.
            # Given 2D, recomputing is cheap and more accurate.
            # However, if we recompute every step, we use 3 calls for J + 1 for F = 4 calls/iter.
            # 100 iters * 4 = 400 calls. Safe.
            J = compute_jacobian(x, Fx)
            if J is None:
                break
        else:
            # Reject step
            lambda_lm = min(lambda_lm * 10.0, lambda_max)
            if lambda_lm > lambda_max:
                # Stuck, try a random restart or break
                break

    return best_x
