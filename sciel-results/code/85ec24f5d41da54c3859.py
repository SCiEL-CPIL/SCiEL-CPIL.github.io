"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid of Expected Improvement (EI) and Probability of Improvement (PI),
    weighted by a temperature that adapts to the current best value and uncertainty,
    all normalized by cost to favor cheap, promising points.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-12), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-9)
    
    # Standardize the improvement relative to f_best
    # Use a robust scale based on the spread of mu
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    if mu_range < 1e-12:
        mu_range = 1e-12
    
    # Temperature parameter: adapts to how close we are to the minimum
    # When f_best is close to 0 (minimum), we want more exploitation
    # When f_best is large, we want more exploration
    # Rosenbrock min is 0, so f_best approaching 0 means we're close
    temp_scale = max(abs(f_best), 1e-6)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # UCB-like term for exploration
    # Use a beta that balances exploration/exploitation
    # Beta decays as we get closer to the optimum
    beta = 0.5 * np.sqrt(np.log(1.0 + np.abs(f_best) / temp_scale + 1e-6))
    ucb = mu + beta * sigma
    
    # Invert UCB since we're minimizing (lower mu is better)
    # We want high scores for low mu, so use -ucb
    ucb_score = -ucb
    
    # Normalize scores to comparable scales
    # Normalize EI
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-12:
        ei_range = 1e-12
    ei_norm = (ei - ei_min) / ei_range
    
    # Normalize PI
    pi_max = np.max(pi)
    pi_min = np.min(pi)
    pi_range = pi_max - pi_min
    if pi_range < 1e-12:
        pi_range = 1e-12
    pi_norm = (pi - pi_min) / pi_range
    
    # Normalize UCB score
    ucb_max = np.max(ucb_score)
    ucb_min = np.min(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-12:
        ucb_range = 1e-12
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Weighted combination: adapt weights based on uncertainty
    # When sigma is high, weight UCB more; when low, weight EI more
    sigma_mean = np.mean(sigma)
    sigma_scale = max(sigma_mean, 1e-9)
    
    # Exploration weight increases with average uncertainty
    expl_weight = 0.3 * (1.0 + np.log1p(sigma_scale / temp_scale))
    expl_weight = np.clip(expl_weight, 0.1, 0.8)
    
    # Combine: EI (exploitation), PI (confidence), UCB (exploration)
    score = (1.0 - expl_weight) * (0.7 * ei_norm + 0.3 * pi_norm) + expl_weight * ucb_norm
    
    # Cost-aware: divide by cost, but use a soft penalty to avoid zero division
    # Use cost^alpha with alpha < 1 to soften the penalty
    cost_penalty = np.power(cost, 0.5)
    score = score / cost_penalty
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
