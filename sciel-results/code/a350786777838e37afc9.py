"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a
    quantile-based exploration bonus to encourage systematic coverage
    of high-uncertainty regions.
    
    mu, sigma: shape (2000,)
    f_best: scalar, lowest observed value
    Returns: shape (2000,) scores, higher = evaluate next
    """
    n = len(mu)
    
    # Basic Expected Improvement
    z = (f_best - mu) / (sigma + 1e-12)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Quantile-based exploration: partition candidates into bins by posterior mean,
    # and within each bin, reward high uncertainty (sigma). This forces exploration
    # of diverse regions rather than clustering.
    
    # Sort by mu to create quantile bins
    order = np.argsort(mu)
    n_bins = 20
    bin_size = n // n_bins
    
    scores = np.zeros(n)
    
    for i in range(n_bins):
        bin_start = i * bin_size
        bin_end = bin_start + bin_size if i < n_bins - 1 else n
        bin_indices = order[bin_start:bin_end]
        
        # Within this bin, score based on a combination of EI and sigma
        # Higher sigma in a bin means more exploration value
        bin_mu = mu[bin_indices]
        bin_sigma = sigma[bin_indices]
        
        # Normalize sigma within the bin to [0, 1]
        sigma_min = np.min(bin_sigma)
        sigma_max = np.max(bin_sigma)
        if sigma_max - sigma_min > 1e-12:
            sigma_norm = (bin_sigma - sigma_min) / (sigma_max - sigma_min)
        else:
            sigma_norm = np.ones_like(bin_sigma)
        
        # Combine: 70% EI (normalized) + 30% exploration bonus
        bin_ei = ei[bin_indices]
        ei_min = np.min(bin_ei)
        ei_max = np.max(bin_ei)
        if ei_max - ei_min > 1e-12:
            ei_norm = (bin_ei - ei_min) / (ei_max - ei_min)
        else:
            ei_norm = np.ones_like(bin_ei)
        
        bin_scores = 0.7 * ei_norm + 0.3 * sigma_norm
        scores[bin_indices] = bin_scores
    
    # Add a small global EI component to ensure exploitation still matters
    global_ei_min = np.min(ei)
    global_ei_max = np.max(ei)
    if global_ei_max - global_ei_min > 1e-12:
        global_ei_norm = (ei - global_ei_min) / (global_ei_max - global_ei_min)
    else:
        global_ei_norm = np.ones_like(ei)
    
    # Final score: blend local quantile-based score with global EI
    final_scores = 0.6 * scores + 0.4 * global_ei_norm
    
    # Ensure all scores are finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_scores
# EVOLVE-BLOCK-END
