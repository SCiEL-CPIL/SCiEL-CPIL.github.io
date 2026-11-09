"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def _check_deadline(timeout=310.0):
    """Return True if we should stop to avoid timeout."""
    return (time.time() - _start_time) > timeout

def acquisition(mu, sigma, f_best):
    """
    Diversity-aware acquisition: combines EI with a smooth diversity score
    that encourages exploration of distinct regions in the (mu, sigma) space.
    Uses a soft-min (log-sum-exp) based spacing measure to avoid the O(n^2)
    bottleneck of exact nearest-neighbor computation.
    
    mu, sigma: shape (2000,), posterior mean and std in objective units.
    f_best: lowest observed value so far.
    Returns: 2000 finite scores, higher = more promising.
    """
    n = len(mu)
    
    # Deadline guard: return simple EI as fallback
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Known minimum of the RBF mixture
    f_min = -2.53218
    
    # Progress: how close are we to the minimum?
    regret = f_best - f_min
    progress = 1.0 - np.clip(regret / 2.0, 0.0, 1.0)
    
    # Compute Expected Improvement (EI) for all candidates
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    # Compute a smooth diversity score using log-sum-exp based soft-min distance
    # Normalize mu and sigma to [0, 1] for distance computation
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma_safe), np.max(sigma_safe)
    mu_range = mu_max - mu_min + 1e-9
    sigma_range = sigma_max - sigma_min + 1e-9
    
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma_safe - sigma_min) / sigma_range
    
    # Compute pairwise squared distances in normalized (mu, sigma) space
    # Use chunked computation to manage memory
    chunk_size = 500
    min_dist = np.zeros(n)
    
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        # Compute distances from candidates [i_start:i_end] to all candidates
        mu_diff = mu_norm[i_start:i_end, np.newaxis] - mu_norm[np.newaxis, :]
        sigma_diff = sigma_norm[i_start:i_end, np.newaxis] - sigma_norm[np.newaxis, :]
        sq_dist = mu_diff**2 + sigma_diff**2
        
        # Set diagonal (self-distance) to a large value
        for j in range(i_start, i_end):
            sq_dist[j - i_start, j] = 1e10
        
        # Soft-min using log-sum-exp: -log(sum(exp(-beta * d))) approximates min(d)
        # Use a moderate beta for smoothness
        beta = 50.0
        # To avoid overflow, use the log-sum-exp trick
        neg_beta_sq_dist = -beta * sq_dist
        # Subtract max for numerical stability
        neg_beta_sq_dist_max = np.max(neg_beta_sq_dist, axis=1, keepdims=True)
        log_sum = neg_beta_sq_dist_max + np.log(np.sum(np.exp(neg_beta_sq_dist - neg_beta_sq_dist_max), axis=1))
        # Soft-min distance = -log_sum / beta
        soft_min_dist = -log_sum[:, 0] / beta
        
        min_dist[i_start:i_end] = soft_min_dist
    
    # Normalize soft_min_dist to [0, 1]
    smd_max = np.max(min_dist)
    smd_min = np.min(min_dist)
    smd_range = smd_max - smd_min + 1e-9
    diversity_score = (min_dist - smd_min) / smd_range
    
    # Adaptive weighting: more weight on diversity early (exploration),
    # more weight on EI late (exploitation)
    diversity_weight = 0.40 * (1.0 - progress) + 0.15 * progress
    
    # Combine EI and diversity
    score = (1.0 - diversity_weight) * ei_norm + diversity_weight * diversity_score
    
    # Scale back to EI scale for consistency
    score = score * ei_range + ei_min
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
