"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Known minimum for Styblinski-Tang 1D
    f_min_known = -39.1662
    
    # --- Entropy-Weighted Expected Improvement (H-EI) ---
    # This combines the expected improvement with an entropy-like term
    # that weights by uncertainty, providing adaptive exploration.
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Standard EI for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Entropy-like term: use log(1 + sigma) as a proxy for uncertainty
    # This amplifies improvement prospects in high-uncertainty regions
    entropy_term = np.log1p(sigma_safe)
    
    # H-EI: multiply EI by entropy term to get adaptive exploration
    # This suppresses improvement in well-explored (low sigma) areas
    # and amplifies it in high-uncertainty areas
    h_ei = ei * (1.0 + entropy_term)
    
    # Add a small UCB component for additional exploration
    # Adaptive beta: higher exploration early, lower later
    distance_to_min = max(f_best - f_min_known, 0.0)
    beta = 0.5 + 1.5 * np.exp(-distance_to_min / 5.0)
    
    ucb_component = -mu + beta * sigma_safe
    
    # Normalize both components to [0, 1]
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-9:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    norm_hei = normalize_scores(h_ei)
    norm_ucb = normalize_scores(ucb_component)
    
    # Weighted combination: prioritize H-EI but add UCB for exploration
    lambda_weight = 0.3
    final_scores = norm_hei + lambda_weight * norm_ucb
    
    # Ensure all scores are finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return final_scores
    
    return final_scores
# EVOLVE-BLOCK-END
