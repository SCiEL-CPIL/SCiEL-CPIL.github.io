import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
        n = len(mu)
        scores = np.zeros(n)
        
        # Number of quantile bins
        k_bins = 20
        
        # Compute quantile indices
        # We want to sort by mu (lower is better)
        # Bin 0: best 1/20, Bin 19: worst 1/20
        quantiles = np.linspace(0, 1, k_bins + 1)
        bin_edges = np.quantile(mu, quantiles)
        
        # Assign bin index to each point
        # np.digitize returns 1..k_bins
        # We want 0..k_bins-1
        bin_indices = np.digitize(mu, bin_edges[1:-1]) - 1
        # Clip to handle edge cases (e.g., min/max values)
        bin_indices = np.clip(bin_indices, 0, k_bins - 1)
        
        # For each bin, find the index with the maximum sigma
        # We also want to keep the global best mu point
        
        # 1. Identify global best mu index
        i_best_mu = np.argmin(mu)
        
        # 2. For each bin, find max sigma index
        # We can do this by iterating or using advanced indexing
        # Since k_bins is small, iteration is fine
        
        max_sigma_indices = []
        for b in range(k_bins):
            mask = (bin_indices == b)
            if np.any(mask):
                indices_in_bin = np.where(mask)[0]
                sigmas_in_bin = sigma[indices_in_bin]
                i_max_sigma_local = np.argmax(sigmas_in_bin)
                i_global = indices_in_bin[i_max_sigma_local]
                max_sigma_indices.append(i_global)
        
        # Assign scores
        # Base score 0
        
        # Score for global best mu
        # We want this to be competitive.
        # If we just set it to a high constant, it might always be picked,
        # which is good for exploitation.
        scores[i_best_mu] = 1000.0 + (f_best - mu[i_best_mu]) # Higher if mu is much lower than f_best?
        # Actually, just a high constant is safer to ensure it's considered.
        # But we want the *highest* score to be picked.
        # Let's make the explore points also high.
        
        # Score for max sigma in each bin
        # We want to encourage exploration.
        # Score = 500.0 + sigma[i] ?
        # If sigma is large, score is higher.
        
        for i in max_sigma_indices:
            # If this index is the same as i_best_mu, we already assigned a high score.
            # We can take the max.
            current_score = 500.0 + sigma[i]
            if current_score > scores[i]:
                scores[i] = current_score
        
        # To ensure we don't always pick the same explore point,
        # we could add some noise? No, deterministic is better for reproducibility in tests.
        
        # What about points that are not in these top sets?
        # They have score 0.
        # The argmax will be one of the ~21 points.
        
        return scores
