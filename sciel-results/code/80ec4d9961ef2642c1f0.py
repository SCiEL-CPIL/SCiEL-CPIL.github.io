"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Thompson Sampling variant for multimodal minimization.
    
    Samples from the posterior predictive distribution to balance exploration
    and exploitation stochastically. The score is based on sampled values
    adjusted by an exploitation bias toward lower predicted means.
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Sample from the posterior predictive: y = mu - sigma * z where z ~ N(0,1)
    # For minimization, we want to select points with low sampled values
    # So higher score means lower sampled value
    
    # Generate random samples
    z = np.random.standard_normal(len(mu))
    samples = mu - sigma_safe * z
    
    # The score should be higher for points with lower sampled values
    # So we negate the samples: score = -samples
    # But we also want a deterministic exploitation bias toward points
    # with low mu (good predictions)
    
    # Normalize samples to get a ranking
    sample_min = np.min(samples)
    sample_max = np.max(samples)
    
    if sample_max - sample_min > eps:
        # Normalize to [0, 1], where 1 is best (lowest sample)
        sample_norm = (sample_max - samples) / (sample_max - sample_min)
    else:
        sample_norm = np.ones_like(samples)
    
    # Exploitation bias: favor points with low mu
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    
    if mu_max - mu_min > eps:
        mu_norm = (mu_max - mu) / (mu_max - mu_min)  # 1 is best (lowest mu)
    else:
        mu_norm = np.ones_like(mu)
    
    # Blend: primarily Thompson sampling with some exploitation bias
    # Weight the stochastic component heavily but add deterministic guidance
    weight_ts = 0.7
    weight_exploit = 0.3
    
    score = weight_ts * sample_norm + weight_exploit * mu_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
