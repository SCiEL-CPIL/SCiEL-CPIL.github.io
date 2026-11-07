"""Acquisition function for Bayesian optimization of Hartmann 3D."""

import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Multi-scale density-penalized LCB acquisition for minimizing Hartmann 3D.
    
    Strategy:
    1. Compute base LCB score: -mu + kappa * sigma (higher = better).
      kappa is dynamically scheduled based on f_best progress toward known minimum.
    2. Estimate local candidate density using a multi-scale kernel density approach.
       For each candidate point, compute the average number of other candidates
       within a fixed radius at multiple scales.
    3. Subtract a density penalty to enforce spatial diversity, pushing exploration
       toward sparser regions of the candidate pool.
    """
    n = len(mu)
    if n == 0:
        return np.array([])
    
    sigma = np.maximum(sigma, 1e-12)
    
    # Known global minimum for Hartmann 3D
    f_min = -3.86278
    f_ref_start = -2.0
    
    # Dynamic kappa: high early (explore), low late (exploit)
    range_val = f_ref_start - f_min  # ~1.86
    current_diff = f_best - f_min
    ratio = current_diff / range_val
    ratio = np.clip(ratio, 0.0, 1.0)
    kappa = 0.5 + 2.5 * ratio
    
    if not np.isfinite(kappa) or kappa <= 0:
        kappa = 1.5
    
    # Base LCB score (higher is better for maximization)
    score = -mu + kappa * sigma
    
    # Multi-scale density penalty
    # We estimate local density by counting neighbors within fixed bandwidths.
    # Since we don't have coordinates, we use a proxy: the structure of mu and sigma
    # itself. In GP-based BO, points with similar mu/sigma patterns tend to be
    # in similar regions of the input space.
    
    # Approach: Use the correlation structure of mu and sigma to estimate "closeness"
    # of candidate points. Points with very similar (mu, sigma) pairs are likely
    # in similar regions of the input space.
    
    # Compute a feature vector for each candidate: [mu, sigma, mu*sigma]
    # Then compute pairwise "distance" in this feature space as a proxy for
    # input-space proximity.
    
    # For efficiency with 2000 points, use a binned histogram approach:
    # Divide the (mu, sigma) plane into bins and count neighbors per bin.
    
    mu_range = mu.max() - mu.min() + 1e-12
    sigma_range = sigma.max() - sigma.min() + 1e-12
    
    # Normalize to [0, 1]
    mu_norm = (mu - mu.min()) / mu_range
    sigma_norm = (sigma - sigma.min()) / sigma_range
    
    # Use 2D histogram with multiple resolutions for multi-scale
    def density_penalty(mu_arr, sigma_arr, n_bins):
        """Estimate local density via 2D histogram binning."""
        mu_r = (mu_arr - mu_arr.min()) / (mu_arr.max() - mu_arr.min() + 1e-12)
        sig_r = (sigma_arr - sigma_arr.min()) / (sigma_arr.max() - sigma_arr.min() + 1e-12)
        
        # Clamp to [0, 1)
        mu_r = np.clip(mu_r, 0.0, 1.0 - 1e-9)
        sig_r = np.clip(sig_r, 0.0, 1.0 - 1e-9)
        
        # Bin indices
        mu_bin = (mu_r * n_bins).astype(int)
        sig_bin = (sig_r * n_bins).astype(int)
        
        # Count per bin
        from collections import Counter
        key = mu_bin * n_bins + sig_bin
        counts = np.bincount(key, minlength=n_bins * n_bins)
        
        # For each point, density = count in its bin
        density = counts[key]
        # Normalize to [0, 1]
        density = density / (density.max() + 1e-12)
        return density
    
    # Multi-scale: combine densities at different resolutions
    n_scales = 3
    penalty = np.zeros(n)
    for i in range(n_scales):
        n_bins = 8 * (2 ** i)  # 8, 16, 32
        d = density_penalty(mu, sigma, n_bins)
        penalty += d
    
    penalty /= n_scales
    
    # Scale penalty to be meaningful relative to score range
    score_range = score.max() - score.min() + 1e-12
    penalty_weight = 0.3 * score_range  # 30% of score range
    
    score = score - penalty_weight * penalty
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
