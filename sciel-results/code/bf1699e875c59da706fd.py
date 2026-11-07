"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time since module load for deadline guard
_MODULE_START = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 20.0


def _time_remaining():
    elapsed = time.time() - _MODULE_START
    return _DEADLINE - elapsed - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a
    quantile-based exploration bonus to encourage systematic coverage
    of high-uncertainty regions.
    
    mu, sigma: shape (2000,)
    f_best: scalar, lowest observed value
    Returns: shape (2000,) scores, higher = evaluate next
    """
    # Time guard: if we're running out of time, return a simple EI score
    if _time_remaining() < 0.5:
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Basic Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Quantile-based exploration: partition candidates into bins by posterior mean,
    # and within each bin, reward high uncertainty (sigma). This forces exploration
    # of diverse regions rather than clustering.
    
    # Sort by mu to create quantile bins
    order = np.argsort(mu)
    n_bins = 10
    bin_size = n // n_bins
    
    scores = np.zeros(n)
    
    for i in range(n_bins):
        bin_start = i * bin_size
        bin_end = bin_start + bin_size if i < n_bins - 1 else n
        bin_indices = order[bin_start:bin_end]
        
        # Within this bin, score based on a combination of EI and sigma
        # Higher sigma in a bin means more exploration value
        bin_mu = mu[bin_indices]
        bin_sigma = sigma_safe[bin_indices]
        
        # Normalize sigma within the bin to [0, 1]
        sigma_min = np.min(bin_sigma)
        sigma_max = np.max(bin_sigma)
        if sigma_max - sigma_min > 1e-12:
            sigma_norm = (bin_sigma - sigma_min) / (sigma_max - sigma_min)
        else:
            sigma_norm = np.ones_like(bin_sigma)
        
        # Combine: 80% EI (normalized) + 20% exploration bonus
        bin_ei = ei[bin_indices]
        ei_min = np.min(bin_ei)
        ei_max = np.max(bin_ei)
        if ei_max - ei_min > 1e-12:
            ei_norm = (bin_ei - ei_min) / (ei_max - ei_min)
        else:
            ei_norm = np.ones_like(bin_ei)
        
        bin_scores = 0.8 * ei_norm + 0.2 * sigma_norm
        scores[bin_indices] = bin_scores
    
    # Add a global EI component to ensure exploitation still matters
    global_ei_min = np.min(ei)
    global_ei_max = np.max(ei)
    if global_ei_max - global_ei_min > 1e-12:
        global_ei_norm = (ei - global_ei_min) / (global_ei_max - global_ei_min)
    else:
        global_ei_norm = np.ones_like(ei)
    
    # Final score: blend local quantile-based score with global EI
    final_scores = 0.7 * scores + 0.3 * global_ei_norm
    
    # Ensure all scores are finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_scores
# EVOLVE-BLOCK-END
