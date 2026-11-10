# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Precompute projection indices for x, y, x-y, x+y, x+2y.
    2. Use Dinkelbach's algorithm to transform the ratio maximization 
       problem max_p D(p)/M(p) into a sequence of convex optimization problems:
       max_p D(p) - t*M(p) subject to p in Simplex.
    3. Solve the convex sub-problems using SLSQP with gradient information.
       Since D and M are concave (M is max of concave, hence concave), 
       D - t*M is concave. Maximizing a concave function over a convex set
       is a convex optimization problem, ensuring global optimality for the sub-problem.
    4. Iterate t_{k+1} = D(p_k)/M(p_k) until convergence.
    """
    n = 10
    deadline = time.time() + 590.0  # Safety margin (limit is 630s)
    
    # Grid coordinates
    # P[y, x] -> y is row, x is col
    ys, xs = np.mgrid[0:n, 0:n]
    
    # Flatten for binning
    # Note: The problem defines P[y, x]. 
    d_val = (xs - ys).flatten()
    m0_val = xs.flatten()
    m1_val = (xs + ys).flatten()
    m2_val = (xs + 2 * ys).flatten()
    
    # Unique values and inverse mapping
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)
    
    num_d = len(d_vals)
    num_m0 = len(m0_vals)
    num_m1 = len(m1_vals)
    num_m2 = len(m2_vals)
    
    # Precompute bin indices as integer arrays for fast indexing
    # We will compute bin probabilities via matrix multiplication or bincount
    # For gradients, we need to know which cell contributes to which bin.
    
    def compute_distributions(p_flat):
        """
        Returns p_d, p_m0, p_m1, p_m2
        """
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)
        return p_d, p_m0, p_m1, p_m2

    def compute_entropies_and_grads(p_flat):
        """
        Computes D, M and their gradients with respect to p_flat.
        p_flat: shape (100,)
        
        Returns:
        D: float
        M: float
        grad_D: shape (100,)
        grad_M: shape (100,)
        """
        p_d, p_m0, p_m1, p_m2 = compute_distributions(p_flat)
        
        def entropy_and_grad(p_dist, inv_map, num_bins):
            # p_dist: shape (num_bins,)
            # inv_map: shape (100,)
            # We need grad_p = dH/dp_flat
            # H = - sum p_dist log2 p_dist
            # dH/dp_bin = -log2(p_bin) - 1/ln(2)
            # dH/dp_flat_i = sum_{bins b where inv_map[i]=b} (dH/dp_bin)
            
            # Handle zeros for log
            p_nonzero = p_dist > 1e-15
            # Gradient with respect to bin probabilities
            # g_bin[b] = -log2(p_dist[b]) - 1/ln(2) if p_dist[b] > 0 else 0
            # Note: mathematically, the gradient is undefined at 0, but we can set it to 0 or a large negative number.
            # Since we are maximizing concave function, and boundary is handled by simplex constraint,
            # setting grad to 0 for p=0 is a standard heuristic for interior-point or penalty methods, 
            # but for SLSQP on simplex, it's safer to handle carefully.
            
            g_bin = np.zeros_like(p_dist)
            if np.any(p_nonzero):
                # log2(p) = log(p) / ln(2)
                g_bin[p_nonzero] = -np.log2(p_dist[p_nonzero]) - 1.0 / np.log(2)
            
            # Map back to flat indices
            # grad_flat[i] = g_bin[inv_map[i]]
            grad_flat = g_bin[inv_map]
            
            H = -np.sum(p_dist[p_nonzero] * np.log2(p_dist[p_nonzero]))
            return H, grad_flat

        H_d, grad_d = entropy_and_grad(p_d, d_inv, num_d)
        H_m0, grad_m0 = entropy_and_grad(p_m0, m0_inv, num_m0)
        H_m1, grad_m1 = entropy_and_grad(p_m1, m1_inv, num_m1)
        H_m2, grad_m2 = entropy_and_grad(p_m2, m2_inv, num_m2)
        
        M = max(H_m0, H_m1, H_m2)
        
        # Gradient of M = max(H_m0, H_m1, H_m2)
        # Subgradient is the gradient of the active term(s).
        # For optimization, we pick the active one (or average if multiple).
        # If multiple are equal, the subdifferential is the convex hull. 
        # Picking one is sufficient for SLSQP as it handles non-smoothness via subgradients.
        
        if M < 1e-9:
            return 0.0, 0.0, np.zeros(100), np.zeros(100)
            
        if H_m0 == M:
            grad_M = grad_m0
        elif H_m1 == M:
            grad_M = grad_m1
        else:
            grad_M = grad_m2
            
        # If there are ties, we could average, but picking the max is standard for max functions.
        # To be more robust against numerical noise in ties:
        # Let's check if they are close.
        max_val = M
        # Simple approach: just use the one that is strictly max, or first max found.
        # The above if/elif/else logic picks the first one found in order m0, m1, m2.
        
        return H_d, M, grad_d, grad_M

    # Dinkelbach's Algorithm
    # Start with t = 0
    t = 0.0
    p_best = np.ones(100) / 100.0
    val_best = 0.0
    
    # Initial guess for p
    p_current = np.ones(100) / 100.0
    
    max_iter = 50
    tol = 1e-6
    
    for k in range(max_iter):
        if time.time() > deadline - 10:
            break
            
        # Objective: max_p D(p) - t * M(p)
        # Subject to: sum(p) = 1, p >= 0
        
        def objective_dinkelbach(p_flat):
            """
            Returns negative of (D - t*M) and its gradient
            """
            D, M, grad_D, grad_M = compute_entropies_and_grads(p_flat)
            val = D - t * M
            # Gradient of -val = -grad_D + t * grad_M
            grad = -grad_D + t * grad_M
            return -val, grad

        # Bounds: p_i >= 0
        bounds = [(0.0, 1.0) for _ in range(100)]
        
        # Constraint: sum(p) = 1
        def constraint_sum(p_flat):
            return np.sum(p_flat) - 1.0
        
        def constraint_sum_grad(p_flat):
            return np.ones(100)
            
        cons = ({'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad})
        
        try:
            # Use SLSQP as it handles constraints well
            res = minimize(
                objective_dinkelbach, 
                p_current, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success:
                p_new = res.x
                # Ensure non-negative and normalized
                p_new = np.clip(p_new, 0.0, None)
                sum_p = np.sum(p_new)
                if sum_p > 1e-9:
                    p_new /= sum_p
                else:
                    p_new = p_current
                    
                D_new, M_new, _, _ = compute_entropies_and_grads(p_new)
                if M_new < 1e-9:
                    break
                    
                ratio_new = D_new / M_new
                
                if ratio_new > val_best:
                    val_best = ratio_new
                    p_best = p_new.copy()
                    
                # Check convergence of t
                if abs(ratio_new - t) < tol:
                    break
                else:
                    t = ratio_new
                    p_current = p_new
            else:
                # If optimization fails, break to avoid infinite loop
                break
                
        except Exception as e:
            # If exception, break
            break

    # Final result is p_best
    P = p_best.reshape((n, n))
    
    # Final validation
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P

# EVOLVE-BLOCK-END
