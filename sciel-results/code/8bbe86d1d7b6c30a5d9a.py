import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Strategy:
    1. Define a family of heuristics based on tightness of fit (inverse waste power)
       and size similarity/product terms.
    2. Use a fast greedy simulation proxy to evaluate the quality of each candidate
       heuristic. The proxy simulates the ACO's decision process by greedily picking
       the item with the highest heuristic value that fits in the current bin.
    3. Select the heuristic that minimizes the number of bins in the proxy simulation.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    cap = float(capacity)
    
    if n == 0:
        return np.empty((0, 0))
    
    # Precompute pairwise sums
    pair_sum = d[:, None] + d[None, :]
    fits = pair_sum <= cap
    residual = cap - pair_sum
    
    # Ensure residual is non-negative where it fits
    residual_safe = np.where(fits, residual, 1.0)
    
    # Precompute size difference for similarity bonus
    size_diff = np.abs(d[:, None] - d[None, :])
    
    # Precompute product
    products = d[:, None] * d[None, :]
    
    def compute_heuristic(gamma, use_product=False, use_similarity=False, sim_weight=0.1):
        """
        Compute heuristic with given parameters.
        gamma: exponent for inverse waste term
        use_product: multiply by product of sizes
        use_similarity: multiply by similarity bonus
        """
        # Inverse waste term: 1 / (residual + eps)^gamma
        eps = 1e-6
        base = 1.0 / (residual_safe + eps)
        
        if gamma != 1.0:
            base = base ** gamma
        
        H = base.copy()
        
        if use_product:
            H = H * products
        
        if use_similarity:
            sim_bonus = 1.0 / (1.0 + size_diff / cap)
            H = H * (1.0 + sim_weight * sim_bonus)
        
        # Mask non-fitting pairs
        H = np.where(fits, H, 1e-9)
        
        # Ensure finiteness and positivity
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        
        return H
    
    def greedy_sim(d, H, max_bins=100):
        """
        Fast greedy simulation: 
        Repeatedly open a bin, place a random fitting item, then repeatedly pick
        the fitting item with highest H[current, j] until no item fits.
        Count bins used.
        """
        n = len(d)
        remaining = set(range(n))
        bins_used = 0
        
        # Precompute for speed: for each item, list of items that fit with it
        # Actually, let's just do it naively but efficiently with numpy
        
        while remaining:
            bins_used += 1
            if bins_used > max_bins:
                return bins_used
            
            # Pick a random item from remaining to start the bin
            start = list(remaining)[0]  # Simplified: just take first
            remaining.discard(start)
            
            current = start
            current_sum = d[current]
            
            while True:
                # Find best next item
                best_j = -1
                best_score = -1
                
                for j in remaining:
                    if current_sum + d[j] <= cap:
                        score = H[current, j]
                        if score > best_score:
                            best_score = score
                            best_j = j
                
                if best_j == -1:
                    break
                
                remaining.discard(best_j)
                current = best_j
                current_sum += d[current]
        
        return bins_used
    
    # Candidate parameters to try
    candidates = []
    
    # Vary gamma (inverse waste exponent)
    for gamma in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        candidates.append((gamma, False, False, 0.0))
    
    # With product term
    for gamma in [0.5, 1.0, 1.5, 2.0]:
        candidates.append((gamma, True, False, 0.0))
    
    # With similarity term
    for gamma in [1.0, 1.5, 2.0]:
        for sim_w in [0.05, 0.1, 0.2]:
            candidates.append((gamma, False, True, sim_w))
    
    # With both product and similarity
    for gamma in [1.0, 1.5]:
        for sim_w in [0.05, 0.1]:
            candidates.append((gamma, True, True, sim_w))
    
    best_H = None
    best_bins = float('inf')
    
    for gamma, use_prod, use_sim, sim_w in candidates:
        if time.time() > deadline - 30:
            break
        
        H = compute_heuristic(gamma, use_prod, use_sim, sim_w)
        
        # Evaluate with greedy simulation
        bins = greedy_sim(d, H)
        
        if bins < best_bins:
            best_bins = bins
            best_H = H
    
    # Fallback if no candidate was evaluated or all failed
    if best_H is None:
        best_H = compute_heuristic(1.0)
    
    # Final safety checks
    if best_H.shape != (n, n):
        best_H = np.full((n, n), 1e-9)
    
    best_H = np.where(np.isfinite(best_H), best_H, 1e-9)
    best_H[best_H < 0] = 1e-9
    
    return best_H
