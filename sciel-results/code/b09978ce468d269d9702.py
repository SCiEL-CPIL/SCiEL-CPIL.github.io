import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a blend of Expected Improvement and UCB, adapting the balance
    based on how close f_best is to the known minimum (0).
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB score for minimization: -mu + beta * sigma
    # Choose beta adaptively based on f_best
    # If f_best is close to 0 (good), exploit more (lower beta)
    # If f_best is far from 0 (bad), explore more (higher beta)
    # f_best for Weierstrass on [-0.5, 0.5] ranges from 0 to ~2
    # Use a smooth transition
    f_best_clamped = np.clip(f_best, 0, 2)
    # beta from 0.5 (exploit) to 2.0 (explore)
    beta = 0.5 + 1.5 * (1.0 - np.exp(-2.0 * f_best_clamped))
    
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize both to have zero mean and unit variance for blending
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-12
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score) + 1e-12
    ucb_norm = (ucb_score - ucb_mean) / ucb_std
    
    # Blend: weight EI more when f_best is good (close to 0)
    # weight_ei goes from 0.3 (explore) to 0.9 (exploit)
    weight_ei = 0.3 + 0.6 * np.exp(-1.5 * f_best_clamped)
    weight_ucb = 1.0 - weight_ei
    
    score = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
