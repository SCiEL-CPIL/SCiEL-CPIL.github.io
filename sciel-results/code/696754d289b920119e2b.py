# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = 620  # safety margin
    
    def compute_P(support, weights):
        """Compute P[X1+X2+X3 < 2*X4] exactly."""
        n = len(support)
        # Normalize weights
        w = weights / weights.sum()
        s = support
        
        # Pairwise sums s_i + s_j
        pair_sums = s[:, None] + s[None, :]
        
        # For each (k, l) pair (X3=s_k, X4=s_l), compute threshold = 2*s_l - s_k
        # Then P[X1+X2 < threshold] = sum over (i,j) with pair_sums[i,j] < threshold of w_i * w_j
        # 
        # P = sum_k sum_l w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l-s_k} w_i * w_j)
        
        # Build matrix of weights for pair sums
        pair_weights = w[:, None] * w[None, :]  # w_i * w_j
        
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * s[l] - s[k]
                # Sum of w_i*w_j where s_i + s_j < threshold
                mask = pair_sums < threshold
                P += w[k] * w[l] * pair_weights[mask].sum()
        
        return P
    
    def compute_P_fast(support, weights):
        """Faster computation using sorting."""
        n = len(support)
        w = weights / weights.sum()
        s = support
        
        # Sort support and corresponding weights
        order = np.argsort(s)
        s_sorted = s[order]
        w_sorted = w[order]
        
        # Pair sums in sorted order
        # For each pair (i,j), the sum is s_sorted[i] + s_sorted[j]
        # We need to compute for each threshold T: sum of w_i*w_j where s_i+s_j < T
        
        # Precompute all pair sums and their weights
        pair_sums = np.sort(s_sorted[:, None] + s_sorted[None, :]).flatten()
        pair_weights = (w_sorted[:, None] * w_sorted[None, :]).flatten()
        pair_weights = np.sort(pair_weights)  # This doesn't align with sorted pair_sums!
        
        # Better: keep them aligned
        pair_sums = (s_sorted[:, None] + s_sorted[None, :]).flatten()
        pair_weights = (w_sorted[:, None] * w_sorted[None, :]).flatten()
        
        # Sort by pair_sums
        idx = np.argsort(pair_sums)
        pair_sums_sorted = pair_sums[idx]
        pair_weights_sorted = pair_weights[idx]
        cumsum_weights = np.cumsum(pair_weights_sorted)
        
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * s[l] - s[k]
                # Find how many pair sums are < threshold
                count = np.searchsorted(pair_sums_sorted, threshold, side='left')
                if count > 0:
                    prob = cumsum_weights[count - 1]
                else:
                    prob = 0.0
                P += w[k] * w[l] * prob
        
        return P
    
    # Start with a reasonable initial guess
    # Try a 3-point distribution: {0, a, b}
    best_score = -1
    best_support = None
    best_weights = None
    
    # Try various initial configurations
    def try_config(support, weights):
        nonlocal best_score, best_support, best_weights
        try:
            score = compute_P_fast(support, weights)
            if score > best_score:
                best_score = score
                best_support = support.copy()
                best_weights = weights.copy()
        except:
            pass
    
    # Configuration 1: {0, 1, 10}
    try_config(np.array([0.0, 1.0, 10.0]), np.array([0.3, 0.4, 0.3]))
    
    # Configuration 2: {0, 1, 5}
    try_config(np.array([0.0, 1.0, 5.0]), np.array([0.3, 0.4, 0.3]))
    
    # Configuration 3: {0.1, 1, 10}
    try_config(np.array([0.1, 1.0, 10.0]), np.array([0.3, 0.4, 0.3]))
    
    # Configuration 4: uniform on {0,1,2,3,4}
    try_config(np.arange(5, dtype=np.float64), np.ones(5))
    
    # Configuration 5: {0, 2, 8}
    try_config(np.array([0.0, 2.0, 8.0]), np.array([0.3, 0.4, 0.3]))
    
    # Configuration 6: {0, 1, 3, 9}
    try_config(np.array([0.0, 1.0, 3.0, 9.0]), np.array([0.25, 0.25, 0.25, 0.25]))
    
    # Configuration 7: {0, 1, 4, 16}
    try_config(np.array([0.0, 1.0, 4.0, 16.0]), np.array([0.25, 0.25, 0.25, 0.25]))
    
    # Configuration 8: {0.5, 1, 2, 4, 8}
    try_config(np.array([0.5, 1.0, 2.0, 4.0, 8.0]), np.ones(5))
    
    # Configuration 9: {0, 0.5, 1, 2, 4}
    try_config(np.array([0.0, 0.5, 1.0, 2.0, 4.0]), np.ones(5))
    
    # Configuration 10: {0, 1, 2, 4, 8, 16}
    try_config(np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0]), np.ones(6))
    
    # Configuration 11: Geometric-like {0, 1, 2, 4, 8} with decreasing weights
    try_config(np.array([0.0, 1.0, 2.0, 4.0, 8.0]), np.array([0.4, 0.25, 0.15, 0.1, 0.1]))
    
    # Configuration 12: {0, 1, 3, 9, 27}
    try_config(np.array([0.0, 1.0, 3.0, 9.0, 27.0]), np.ones(5))
    
    # Now try to optimize the best configuration
    if best_support is not None:
        n = len(best_support)
        
        def objective(params):
            if time.time() > deadline - 5:
                return -1
            # params: [s1, s2, ..., sn, w1, w2, ..., wn]
            s = params[:n]
            w = params[n:]
            
            # Ensure support is non-negative and sorted
            s = np.maximum(s, 0.0)
            s = np.sort(s)
            
            # Ensure weights are positive
            w = np.maximum(w, 1e-6)
            
            try:
                score = compute_P_fast(s, w)
                return -score
            except:
                return 1e10
        
        x0 = np.concatenate([best_support, best_weights])
        
        # Try Nelder-Mead optimization
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                    'adaptive': True})
            if result.fun < -best_score and time.time() < deadline - 5:
                s_opt = np.sort(np.maximum(result.x[:n], 0.0))
                w_opt = np.maximum(result.x[n:], 1e-6)
                score = compute_P_fast(s_opt, w_opt)
                if score > best_score:
                    best_score = score
                    best_support = s_opt
                    best_weights = w_opt
        except:
            pass
        
        # Try a few more random perturbations
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            s_pert = best_support * (1 + 0.1 * np.random.randn(n))
            s_pert = np.sort(np.maximum(s_pert, 0.0))
            w_pert = best_weights * (1 + 0.1 * np.random.randn(n))
            w_pert = np.maximum(w_pert, 1e-6)
            
            try:
                score = compute_P_fast(s_pert, w_pert)
                if score > best_score:
                    best_score = score
                    best_support = s_pert.copy()
                    best_weights = w_pert.copy()
            except:
                pass
    
    # Try larger support sets
    for n_pts in [7, 8, 10]:
        if time.time() > deadline - 20:
            break
        
        # Exponential spacing
        support_exp = np.logspace(0, 1, n_pts)  # from 1 to 10
        weights_exp = np.ones(n_pts) / n_pts
        try_config(support_exp, weights_exp)
        
        # Linear spacing
        support_lin = np.linspace(0, 10, n_pts)
        weights_lin = np.ones(n_pts) / n_pts
        try_config(support_lin, weights_lin)
        
        # Mix: some at 0, rest exponential
        if n_pts >= 3:
            support_mix = np.concatenate([[0.0], np.logspace(0, 0.8, n_pts-1)])
            weights_mix = np.concatenate([[0.3], np.ones(n_pts-1) * 0.7 / (n_pts-1)])
            try_config(support_mix, weights_mix)
    
    # Final optimization on the best found
    if best_support is not None and time.time() < deadline - 30:
        n = len(best_support)
        
        def objective2(params):
            if time.time() > deadline - 5:
                return -1
            s = np.sort(np.maximum(params[:n], 0.0))
            w = np.maximum(params[n:], 1e-6)
            try:
                return -compute_P_fast(s, w)
            except:
                return 1e10
        
        x0 = np.concatenate([best_support, best_weights])
        
        try:
            result = minimize(objective2, x0, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10,
                                    'adaptive': True})
            if result.fun < -best_score:
                s_opt = np.sort(np.maximum(result.x[:n], 0.0))
                w_opt = np.maximum(result.x[n:], 1e-6)
                score = compute_P_fast(s_opt, w_opt)
                if score > best_score:
                    best_score = score
                    best_support = s_opt
                    best_weights = w_opt
        except:
            pass
    
    # Return the best result
    if best_support is None:
        # Fallback to uniform
        return np.arange(32, dtype=np.float64), np.ones(32, dtype=np.float64) / 32
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
