import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin before 600s limit

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1] # ascending
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        We return 1/norm_sq.
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # New Approach: Coordinate Descent with Adaptive Step Sizes and Global Restart
    # This is a first-order method that is often more efficient for this type of problem
    # than global methods like DE when combined with smart initialization and restarts.
    
    # Initialize with the seed geometric sequence
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    best_log_roots = seed_log_roots.copy()
    best_val = cost_func(best_log_roots)
    
    # Define bounds for log_roots
    bounds = [(-4.0, 10.0)] * 12
    lower_bound = -4.0
    upper_bound = 10.0
    
    # Coordinate Descent with adaptive steps
    def coordinate_descent(log_roots, max_iter=5000, tol=1e-10):
        current_log_roots = log_roots.copy()
        current_val = cost_func(current_log_roots)
        step_sizes = np.ones(12) * 0.1
        
        for i in range(max_iter):
            improved = False
            for j in range(12):
                # Try to adjust the j-th coordinate
                best_local_val = current_val
                best_local_coord = current_log_roots[j]
                
                # Adaptive step: try shrinking and expanding
                step = step_sizes[j]
                for _ in range(10):  # Local search around current point
                    # Try positive step
                    new_coord = current_log_roots[j] + step
                    if new_coord < lower_bound or new_coord > upper_bound:
                        step *= 0.5
                        continue
                    trial_log_roots = current_log_roots.copy()
                    trial_log_roots[j] = new_coord
                    trial_val = cost_func(trial_log_roots)
                    if trial_val < best_local_val:
                        best_local_val = trial_val
                        best_local_coord = new_coord
                        improved = True
                        step_sizes[j] = step * 1.5  # Expand step if improvement
                        break
                    else:
                        step *= 0.5
                        if step < 1e-8:
                            break
                    
                    # Try negative step
                    new_coord = current_log_roots[j] - step
                    if new_coord < lower_bound or new_coord > upper_bound:
                        step *= 0.5
                        continue
                    trial_log_roots = current_log_roots.copy()
                    trial_log_roots[j] = new_coord
                    trial_val = cost_func(trial_log_roots)
                    if trial_val < best_local_val:
                        best_local_val = trial_val
                        best_local_coord = new_coord
                        improved = True
                        step_sizes[j] = step * 1.5  # Expand step if improvement
                        break
                    else:
                        step *= 0.5
                        if step < 1e-8:
                            break
                
                if improved:
                    current_log_roots[j] = best_local_coord
                    current_val = best_local_val
                else:
                    step_sizes[j] = max(step, 1e-8)
            
            if not improved and i > 100:
                break
        
        return current_log_roots, current_val

    # Run coordinate descent from multiple initial points
    initial_points = [
        seed_log_roots.copy(),
        np.log(2.0) + np.arange(12) * np.log(2.0),
        np.log(8.0) + np.arange(12) * np.log(2.0),
        np.log(4.0) + np.arange(12) * np.log(1.5),
        np.log(4.0) + np.arange(12) * np.log(2.5),
        np.log(1.0) + np.arange(12) * np.log(3.0),
        np.log(10.0) + np.arange(12) * np.log(1.2),
        np.log(4.0) + np.random.default_rng(42).uniform(-1, 1, 12) * 0.5,
        np.log(4.0) + np.random.default_rng(123).uniform(-1, 1, 12) * 0.5,
        np.log(4.0) + np.random.default_rng(456).uniform(-1, 1, 12) * 0.5,
    ]
    
    for init_point in initial_points:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        try:
            # Ensure initial point is within bounds
            init_point = np.clip(init_point, lower_bound, upper_bound)
            
            # Run coordinate descent
            result_log_roots, result_val = coordinate_descent(init_point, max_iter=3000)
            
            if result_val < best_val:
                best_val = result_val
                best_log_roots = result_log_roots.copy()
                
        except Exception as e:
            pass
    
    # Final polish with L-BFGS-B
    if time.time() - start_time < deadline - 10:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x.copy()
        except Exception:
            pass

    # Convert back to roots
    best_roots = np.exp(best_log_roots)
    best_roots = np.maximum(best_roots, 1e-6)
    
    return list(best_roots)

