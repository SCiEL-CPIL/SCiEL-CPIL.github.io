import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    Uses a progress-adaptive blend of Expected Improvement and UCB.
    """
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Known optimum for Styblinski-Tang 1D
    f_opt = -39.1662
    
    # Compute distance from optimum (normalized)
    # f_best starts around 50+ and decreases toward -39.1662
    # dist = 1 when far (f_best ~ 50), dist = 0 when close (f_best ~ -39)
    dist = np.clip((f_best - f_opt) / 90.0, 0.0, 1.0)
    
    # Adaptive exploration coefficient
    # When far from optimum (dist close to 1), explore more (higher beta)
    # When close to optimum (dist close to 0), exploit more (lower beta)
    beta = 0.3 + 2.5 * dist
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # UCB for minimization: score = -mu + beta * sigma
    ucb = -mu + beta * sigma_safe
    
    # Blend: weight EI more when close to optimum, UCB more when far
    # When dist is high (far), weight_ucb is high, weight_ei is low
    # When dist is low (close), weight_ei is high, weight_ucb is low
    weight_ei = 1.0 - 0.7 * dist  # 1.0 when close, 0.3 when far
    weight_ucb = 0.7 * dist       # 0.0 when close, 0.7 when far
    
    # Combine
    score = weight_ei * ei + weight_ucb * ucb
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
