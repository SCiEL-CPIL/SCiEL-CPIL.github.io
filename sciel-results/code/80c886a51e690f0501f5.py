"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Quantile-binned exploration scoring for Bayesian optimization.
    
    Strategy:
    - Partition the 2000 candidates into disjoint quantile bins of the posterior mean.
    - Within each bin, assign scores based on the maximum negative standard deviation
      (i.e., highest uncertainty) to force systematic exploration of high-uncertainty regions.
    - This prevents clustering of evaluations seen in standard EI/LCB methods.
    
    We blend this with a mild exploitation term to ensure we still converge.
    """
    n = len(mu)
    
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros(n)
    
    # Clip sigma to avoid division issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement as a baseline exploitation score
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI to [0, 1] for blending
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max > ei_min + 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros(n)
    
    # Quantile-binned exploration:
    # Sort candidates by posterior mean (mu) into quantile bins
    # Within each bin, score by -sigma (higher uncertainty = higher score)
    
    n_bins = 20  # Number of quantile bins
    bin_size = n // n_bins
    remainder = n % n_bins
    
    # Create bin indices
    bin_indices = np.zeros(n, dtype=np.int64)
    idx = 0
    for b in range(n_bins):
        size = bin_size + (1 if b < remainder else 0)
        bin_indices[idx:idx+size] = b
        idx += size
    
    # Sort by mu to assign quantile bins
    sorted_idx = np.argsort(mu)
    
    # Assign each candidate to its quantile bin based on rank
    ranks = np.zeros(n, dtype=np.int64)
    ranks[sorted_idx] = np.arange(n)
    
    # Determine bin for each candidate based on rank
    bin_of_candidate = np.zeros(n, dtype=np.int64)
    for b in range(n_bins):
        start = b * bin_size + min(b, remainder)
        end = start + bin_size + (1 if b < remainder else 0)
        bin_of_candidate[sorted_idx[start:end]] = b
    
    # For each bin, find the max -sigma (i.e., min sigma) and use it as a reference
    # Actually, we want to score within each bin by -sigma to encourage exploration
    # of the most uncertain points in each quantile region
    
    # Compute per-bin exploration score: -sigma normalized within bin
    exploration_scores = np.zeros(n)
    for b in range(n_bins):
        mask = (bin_of_candidate == b)
        if np.sum(mask) == 0:
            continue
        sigmas_in_bin = sigma[mask]
        s_min = np.min(sigmas_in_bin)
        s_max = np.max(sigmas_in_bin)
        if s_max > s_min + 1e-12:
            # Higher sigma gets higher score
            exploration_scores[mask] = (sigmas_in_bin - s_min) / (s_max - s_min)
        else:
            exploration_scores[mask] = 0.5
    
    # Blend exploitation and exploration
    # Weight exploration more heavily to encourage systematic coverage
    alpha = 0.6  # Weight for exploration
    scores = alpha * exploration_scores + (1 - alpha) * ei_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
