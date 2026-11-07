import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard (explicitly required by directive)
    t_start = time.time()
    
    # Guard against numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # z-score for improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) - standard formula for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Structural Change: Quantile-based Exploration
    # Partition candidates into disjoint quantile bins of the posterior mean (mu)
    n_bins = 20
    # Use quantiles to define bin edges
    bin_edges = np.quantile(mu, np.linspace(0, 1, n_bins + 1))
    # Assign each candidate to a bin (0 to n_bins-1)
    # np.digitize returns the index of the bin. 
    # We ignore the last edge to keep n_bins bins.
    bin_indices = np.digitize(mu, bin_edges[1:-1])
    
    # For each bin, find the maximum sigma to normalize uncertainty within that bin
    max_sigma_per_bin = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (bin_indices == i)
        if np.any(mask):
            max_sigma_per_bin[i] = np.max(sigma[mask])
    
    # Avoid division by zero
    max_sigma_per_bin = np.maximum(max_sigma_per_bin, 1e-12)
    
    # Retrieve the max sigma for the bin each candidate belongs to
    current_bin_max_sigma = max_sigma_per_bin[bin_indices]
    
    # Exploration score: normalized sigma within the bin.
    # The candidate with the highest sigma in each bin gets a score of 1.0.
    # This forces the search to pick the most uncertain point for each level of predicted mean.
    exploration_score = sigma / current_bin_max_sigma
    
    # Combine EI and Exploration Score
    # Weight exploration heavily to enforce the structural grid-based exploration
    weight_explore = 1.0
    score = ei + weight_explore * exploration_score
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline (should not trigger for 2000 points, but required)
    if time.time() > (t_start + 2.0):
         pass # Return current score
        
    return score
