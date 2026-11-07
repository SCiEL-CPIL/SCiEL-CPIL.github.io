import numpy as np

def acquisition(mu, sigma, f_best):
    n = len(mu)
    scores = np.zeros(n)
    
    if n == 0:
        return scores

    # Number of quantile bins
    k_bins = 20
    
    # Compute quantile edges
    quantiles = np.linspace(0, 1, k_bins + 1)
    bin_edges = np.quantile(mu, quantiles)
    
    # Assign bin indices
    # Using searchsorted is faster than digitize for large n, but n=2000 is small.
    # digitize is fine.
    bin_indices = np.digitize(mu, bin_edges[1:-1])
    bin_indices = np.clip(bin_indices, 0, k_bins - 1)
    
    # Exploration weight
    beta = 1.0
    
    # For each bin, find the best point according to UCB
    # Vectorized approach:
    # Create a masked array or iterate. Iteration is fine for 20 bins.
    
    for b in range(k_bins):
        mask = (bin_indices == b)
        if not np.any(mask):
            continue
        
        indices_in_bin = np.where(mask)[0]
        # Calculate UCB for these points
        # UCB = -mu + beta * sigma
        # Since we are minimizing, lower mu is better.
        # We want to maximize the score.
        ucb_vals = -mu[indices_in_bin] + beta * sigma[indices_in_bin]
        
        # Find the index with the max UCB
        local_best_idx = np.argmax(ucb_vals)
        global_best_idx = indices_in_bin[local_best_idx]
        best_score = ucb_vals[local_best_idx]
        
        # Assign score
        # To ensure these points are preferred over others (which have 0),
        # we can just assign the UCB value.
        # However, if all UCB values are negative, and some non-bin-leader points have 0,
        # the argmax might pick a non-bin-leader point with score 0.
        # We want to force the selection from the bin leaders.
        # So we should add a large constant or ensure scores are positive/high.
        # Or, we can set non-leader scores to -inf? No, must be finite.
        # Let's shift the scores so the max is positive?
        # Or just add a large constant to the bin leaders?
        
        # If we add a constant, we lose the relative ordering? No, constant shift preserves order.
        # But if we add a constant to *all* bin leaders, they will all be > 0 (assuming constant is large enough).
        # Then argmax will pick the one with the highest UCB among them.
        
        # Let's find the global max UCB to set a baseline?
        # Simpler: Just add a large constant to all bin leaders.
        # But what constant? 
        # The range of UCB can be large.
        
        # Alternative: Assign score = UCB + max_UCB_in_all_bins?
        # That requires two passes.
        
        # Let's just use the UCB value, but ensure that if UCB is negative, 
        # it's still better than 0? 
        # If f_best is 10, mu is 10, sigma is 0. UCB = -10.
        # Other points have score 0.
        # Argmax picks 0 (a non-leader). This is bad.
        
        # So we MUST ensure bin leaders have higher scores than non-leaders.
        # Strategy: 
        # 1. Calculate UCB for all points.
        # 2. Identify bin leaders.
        # 3. Set score for bin leaders to their UCB.
        # 4. Set score for non-leaders to -1e10 (very low).
        # This forces selection from bin leaders.
        
        scores[global_best_idx] = best_score
        
    # Set non-leader scores to a very low value
    # We need to identify which indices were NOT leaders.
    # We can track them.
    
    # Let's rewrite to track leaders
    leaders = np.zeros(n, dtype=bool)
    for b in range(k_bins):
        mask = (bin_indices == b)
        if not np.any(mask):
            continue
        indices_in_bin = np.where(mask)[0]
        ucb_vals = -mu[indices_in_bin] + beta * sigma[indices_in_bin]
        local_best_idx = np.argmax(ucb_vals)
        global_best_idx = indices_in_bin[local_best_idx]
        best_score = ucb_vals[local_best_idx]
        
        scores[global_best_idx] = best_score
        leaders[global_best_idx] = True
        
    # Set non-leaders to -inf (or a very small number)
    scores[~leaders] = -1e10
    
    # Ensure finite
    scores[~np.isfinite(scores)] = -1e10
    
    return scores
