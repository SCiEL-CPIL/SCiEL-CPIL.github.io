import numpy as np
import time
import sys
from scipy.optimize import differential_evolution

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    if np.all(h < 1e-15):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Strategy:
    # 1. Use the sparse support idea from inspiration 1: search over K non-zero elements.
    #    This is effective because the optimal solution for this type of problem is often sparse.
    # 2. Use the smooth soft-max surrogate from inspiration 2 to handle the non-differentiability
    #    of the max in the autoconvolution, allowing for better gradient-based polishing.
    # 3. Combine DE for global search with a local polish step.
    
    # Objective for DE: minimize M/Q where M = max(conv), Q = sum(h^2), with sum(h)=1.
    # This is equivalent to maximizing Q/M.
    
    def _objective_sparse(weights):
        """
        Minimize M/Q for a sparse vector defined by weights at fixed positions.
        weights: array of length K, non-negative.
        """
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0):
            return 1e10
        s = np.sum(w)
        if s < 1e-15:
            return 1e10
        w = w / s
        
        q = np.sum(w**2)
        if q < 1e-15:
            return 1e10
            
        conv = np.convolve(w, w, mode='full')
        m = np.max(conv)
        
        if m < 1e-15:
            return 1e10
            
        return m / q

    def _objective_full(h):
        """
        Minimize M/Q for a full vector h.
        h: array of length N, non-negative.
        """
        h = np.asarray(h, dtype=float)
        if np.any(h < 0):
            return 1e10
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h / s
        
        q = np.sum(h**2)
        if q < 1e-15:
            return 1e10
            
        conv = np.convolve(h, h, mode='full')
        m = np.max(conv)
        
        if m < 1e-15:
            return 1e10
            
        return m / q

    def _soft_max_objective(h, temperature=1.0):
        """
        Smoothed version of _objective_full using soft-max for the max operation.
        """
        h = np.asarray(h, dtype=float)
        if np.any(h < 0):
            return 1e10
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h / s
        
        q = np.sum(h**2)
        if q < 1e-15:
            return 1e10
            
        conv = np.convolve(h, h, mode='full')
        
        if temperature <= 0:
            m = np.max(conv)
        else:
            c_max = np.max(conv)
            # Log-sum-exp trick
            exp_vals = np.exp((conv - c_max) / temperature)
            log_sum_exp = c_max + np.log(np.sum(exp_vals))
            m = temperature * log_sum_exp
            
        if m < 1e-15:
            return 1e10
            
        return m / q

    np.random.seed(42)
    
    # Phase 1: Sparse Support Search (Inspiration 1)
    # Try different sparsities K.
    # The positions of the non-zero elements also need to be optimized.
    # For small K, we can brute force or use a heuristic for positions.
    # However, DE on the values is fast. Let's fix positions to the first K indices for now,
    # or better, let's allow the positions to be part of the search if K is small?
    # Actually, for autoconvolution, the "shape" matters.
    # Let's stick to contiguous blocks for simplicity in the sparse phase, or random subsets.
    # Inspiration 1 just used the first K indices. Let's improve by trying a few random subsets.
    
    max_K_sparse = 8
    
    for K in range(2, max_K_sparse + 1):
        if time.time() > deadline - 20.0:
            break
            
        # Try a few different support sets for this K
        num_supports = 5 if K <= 4 else 2
        for support_idx in range(num_supports):
            if time.time() > deadline - 10.0:
                break
                
            # Generate a random support set
            if support_idx == 0:
                # Contiguous at start
                support = list(range(K))
            else:
                support = np.random.choice(N, K, replace=False).tolist()
                # Sort to keep it consistent? Not strictly necessary for conv, but good for determinism
                support.sort()
                
            # DE over the values at these support positions
            bounds = [(0, 1)] * K
            
            def obj_sparse_wrapper(x, sup=support):
                if time.time() > deadline - 5.0:
                    return 1e10
                # Construct full vector
                h_full = np.zeros(N)
                for i, idx in enumerate(sup):
                    h_full[idx] = x[i]
                # Use the normalized objective
                return _objective_full(h_full)
                
            try:
                res = differential_evolution(
                    obj_sparse_wrapper, 
                    bounds, 
                    maxiter=50, 
                    popsize=10, 
                    tol=1e-5, 
                    seed=42 + K + support_idx,
                    polish=True
                )
                
                if res.fun < 1e9:
                    # Reconstruct h
                    h_candidate = np.zeros(N)
                    for i, idx in enumerate(support):
                        h_candidate[idx] = res.x[i]
                    
                    r = _compute_ratio(h_candidate)
                    if r > best_ratio:
                        best_ratio = r
                        best_h = h_candidate.copy()
                        
            except Exception:
                pass

    # Phase 2: Full Vector Optimization with Smooth Surrogate (Inspiration 2)
    # Use the best sparse solution as a starting point for a full vector optimization.
    # We will use a simple gradient ascent with line search on the soft-max objective.
    
    if best_ratio < 1.0:
        # Fallback initialization
        best_h = np.ones(N) / N
        best_ratio = _compute_ratio(best_h)
        
    h_current = best_h.copy()
    
    # Gradient ascent on soft-max objective
    # We want to maximize R = Q / (S * M). Since S=1 is normalized in the objective we minimize M/Q.
    # Let's directly optimize the soft-max score.
    
    def _soft_max_score(h, temperature):
        h = np.asarray(h, dtype=float)
        s = np.sum(h)
        if s < 1e-15:
            return 0.0
        h_norm = h / s
        q = np.sum(h_norm**2)
        if q < 1e-15:
            return 0.0
        conv = np.convolve(h_norm, h_norm, mode='full')
        c_max = np.max(conv)
        exp_vals = np.exp((conv - c_max) / temperature)
        m = temperature * (c_max + np.log(np.sum(exp_vals)))
        if m < 1e-15:
            return 0.0
        return q / m

    # Approximate gradient of soft-max score
    def _soft_max_grad(h, temperature):
        h = np.asarray(h, dtype=float)
        s = np.sum(h)
        if s < 1e-15:
            return np.zeros_like(h)
        h_norm = h / s
        q = np.sum(h_norm**2)
        if q < 1e-15:
            return np.zeros_like(h)
        
        conv = np.convolve(h_norm, h_norm, mode='full')
        c_max = np.max(conv)
        exp_vals = np.exp((conv - c_max) / temperature)
        weights = exp_vals / np.sum(exp_vals)
        m = temperature * (c_max + np.log(np.sum(exp_vals)))
        
        if m < 1e-15:
            return np.zeros_like(h)
            
        # Score = q / m
        # d(score)/dh_i = (dq/dh_i * m - q * dm/dh_i) / m^2
        # Note: h_norm = h/s. This chain rule is tricky.
        # Let's just use finite differences for the gradient to be safe, as N=50 is small.
        
        eps = 1e-5
        grad = np.zeros_like(h)
        score_0 = _soft_max_score(h, temperature)
        
        for i in range(N):
            h_pert = h.copy()
            h_pert[i] += eps
            score_pert = _soft_max_score(h_pert, temperature)
            grad[i] = (score_pert - score_0) / eps
            
        return grad

    # Annealing temperature
    temps = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2]
    
    for temp in temps:
        if time.time() > deadline - 10.0:
            break
            
        # Gradient ascent
        step_size = 0.1
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
                
            grad = _soft_max_grad(h_current, temp)
            
            # Normalize gradient to avoid huge steps
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            grad = grad / grad_norm
            
            h_new = h_current + step_size * grad
            h_new = np.clip(h_new, 0, None)
            
            score_new = _soft_max_score(h_new, temp)
            score_old = _soft_max_score(h_current, temp)
            
            if score_new > score_old:
                h_current = h_new
                # Update best if actual score improved
                actual_score = _compute_ratio(h_current)
                if actual_score > best_ratio:
                    best_ratio = actual_score
                    best_h = h_current.copy()
            else:
                step_size *= 0.5
                
    # Final Polish with L-BFGS-B on the exact objective
    # Minimize -_compute_ratio(h)
    
    def _neg_ratio(h):
        if time.time() > deadline - 1.0:
            return 1e10
        r = _compute_ratio(h)
        return -r

    try:
        from scipy.optimize import minimize
        res = minimize(
            _neg_ratio, 
            best_h, 
            method='L-BFGS-B', 
            bounds=[(0, None)] * N,
            options={'maxiter': 100}
        )
        if res.fun < -best_ratio:
            best_ratio = -res.fun
            best_h = res.x
    except Exception:
        pass

    # Ensure non-negativity and final check
    best_h = np.clip(best_h, 0, None)
    final_ratio = _compute_ratio(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio

    return list(best_h)

if __name__ == "__main__":
    # Quick test
    h = construct()
    print(_compute_ratio(h))
