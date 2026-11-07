"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Pre-compute constants for speed
_EPS = 1e-9


def _expected_improvement(mu, sigma, f_best):
    """Standard EI acquisition."""
    z = (f_best - mu) / (sigma + _EPS)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)


def _probability_improvement(mu, sigma, f_best):
    """PI acquisition."""
    return norm.cdf((f_best - mu) / (sigma + _EPS))


def _ucb(mu, sigma, f_best, xi=2.0):
    """UCB acquisition (higher is better for score, so we maximize mu + xi*sigma)."""
    # For minimization, we want to find low values.
    # UCB for minimization: minimize mu - xi*sigma
    # But we return scores where higher is better.
    # So score = -mu + xi*sigma
    return -mu + xi * sigma


def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition function that blends EI and PI based on uncertainty.
    
    In early stages (high uncertainty), PI is better to avoid over-exploration.
    In later stages (low uncertainty), EI provides better exploitation.
    
    We use a weighted blend where the weight on PI decreases as uncertainty drops.
    """
    # Compute EI scores
    ei_scores = _expected_improvement(mu, sigma, f_best)
    
    # Compute PI scores
    pi_scores = _probability_improvement(mu, sigma, f_best)
    
    # Compute UCB scores
    ucb_scores = _ucb(mu, sigma, f_best, xi=2.0)
    
    # Adaptive blending based on relative uncertainty
    # When sigma is large relative to the spread of mu, use more PI
    # When sigma is small, use more EI
    sigma_mean = np.mean(sigma)
    mu_spread = np.std(mu) + _EPS
    
    # Uncertainty ratio: how much uncertainty relative to the spread
    # High ratio -> more PI, low ratio -> more EI
    uncertainty_ratio = sigma_mean / (mu_spread + _EPS)
    
    # Clip to [0, 1] for the weight
    # weight_pi is the weight on PI, weight_ei is the weight on EI
    # Use a sigmoid-like transition
    # When uncertainty_ratio is high, weight_pi is high
    weight_pi = 1.0 / (1.0 + np.exp(-2.0 * (uncertainty_ratio - 1.0)))
    weight_ei = 1.0 - weight_pi
    
    # Normalize scores to [0, 1] range for fair blending
    def normalize_scores(scores):
        s_min = np.min(scores)
        s_max = np.max(scores)
        s_range = s_max - s_min
        if s_range < _EPS:
            return np.ones_like(scores)
        return (scores - s_min) / s_range
    
    ei_norm = normalize_scores(ei_scores)
    pi_norm = normalize_scores(pi_scores)
    ucb_norm = normalize_scores(ucb_scores)
    
    # Blend EI and PI
    blended = weight_pi * pi_norm + weight_ei * ei_norm
    
    # Add a small UCB component for exploration
    # This helps in early stages when the posterior is poorly constrained
    ucb_weight = 0.1 * weight_pi  # More UCB when uncertainty is high
    blended = (1.0 - ucb_weight) * blended + ucb_weight * ucb_norm
    
    return blended
# EVOLVE-BLOCK-END
