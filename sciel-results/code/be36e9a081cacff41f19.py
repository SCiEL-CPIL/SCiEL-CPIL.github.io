# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _evaluate_measure(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] for the given discrete measure.
    support: (n,) array of non-negative floats
    weights: (n,) array of non-negative floats (will be normalized)
    Returns: float P
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    
    # Compute all pairwise sums S_ij = s_i + s_j
    # S has shape (n, n)
    S = support[:, None] + support[None, :]
    S_flat = S.flatten()
    
    # Weighted pair probabilities
    w_pair = w[:, None] * w[None, :]
    w_pair_flat = w_pair.flatten()
    
    # T[k, l] = 2*s_l - s_k
    T = 2.0 * support[None, :] - support[:, None]
    
    P = 0.0
    # Vectorize the inner sum over i,j for all k,l at once?
    # We can't easily broadcast S_flat (n^2,) against T (n, n) to get a 3D array of indicators (n, n, n^2) without huge memory.
    # But n is small.
    # Let's try to be smart:
    # Sort S_flat and compute weights of sorted S_flat? No, weights are not sorted with S.
    
    # Let's just do the loop. It's simple and robust for small n.
    for k in range(n):
        for l in range(n):
            target = T[k, l]
            # Strict inequality
            mask = S_flat < target
            prob_kl = np.dot(w_pair_flat, mask)
            P += w[k] * w[l] * prob_kl
            
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Optimize the discrete measure using stochastic hill-climbing with random restarts.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin: 10s buffer before 600s kill
    
    best_score = -np.inf
    best_support = None
    best_weights = None
    
    # We will try different numbers of points n
    # Based on insights, n=3 to 6 is promising.
    # Uniform on {0,1,...,n-1} is a decent baseline.
    # The 2-point optimum is around P=0.261.
    # We want to beat that.
    
    # Strategy:
    # 1. Initialize with several different n values.
    # 2. For each, perform local search (perturb support and weights).
    # 3. Keep the global best.
    
    random.seed(42)
    np.random.seed(42)
    
    candidate_ns = [4, 5, 6]
    
    for n in candidate_ns:
        # Multiple restarts for each n
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize support and weights
            # Support: random points in [0, 10] or structured?
            # Let's try mixing: some random, some structured.
            # Structured: 0, 1, 2, ...
            # Random: sorted random in [0, 5]
            
            if r % 2 == 0:
                # Structured start
                support = np.arange(n, dtype=np.float64)
            else:
                # Random start
                support = np.sort(np.random.uniform(0, 5, n))
            
            # Weights: random positive, normalized
            weights = np.random.dirichlet(np.ones(n))
            
            # Local search
            current_score = _evaluate_measure(support, weights)
            if current_score > best_score:
                best_score = current_score
                best_support = support.copy()
                best_weights = weights.copy()
            
            # Iterate local search
            # We will perturb support and weights.
            # Support perturbations: add Gaussian noise.
            # Weight perturbations: add Gaussian noise to log-weights, then softmax.
            
            # Convert weights to log-space for perturbation
            log_w = np.log(weights + 1e-15)
            
            # Local search loop
            local_iter = 200
            for it in range(local_iter):
                if time.time() > deadline:
                    break
                
                # Generate perturbation
                # Scale of perturbation decreases with iterations?
                # Or fixed scale?
                # Let's use adaptive scale.
                scale_s = 0.5 * (0.95 ** (it // 10))
                scale_w = 0.5 * (0.95 ** (it // 10))
                
                # Perturb support
                new_support = support + np.random.normal(0, scale_s, n)
                # Ensure non-negative
                new_support = np.maximum(new_support, 0.0)
                
                # Perturb weights
                new_log_w = log_w + np.random.normal(0, scale_w, n)
                # Softmax
                max_log_w = np.max(new_log_w)
                new_w_unnorm = np.exp(new_log_w - max_log_w)
                new_weights = new_w_unnorm / np.sum(new_w_unnorm)
                
                new_score = _evaluate_measure(new_support, new_weights)
                
                if new_score > current_score:
                    current_score = new_score
                    support = new_support
                    weights = new_weights
                    log_w = new_log_w
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_support = support.copy()
                        best_weights = weights.copy()
                else:
                    # If no improvement, we might still want to occasionally jump?
                    # Hill climbing usually rejects. But to escape local optima, 
                    # the restarts handle that.
                    pass
            
            # Check time
            if time.time() > deadline:
                break
    
    # If we didn't find anything (shouldn't happen), fallback
    if best_support is None:
        n = 4
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    # Final validity check
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
        n = 4
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    return best_support, best_weights
# EVOLVE-BLOCK-END
