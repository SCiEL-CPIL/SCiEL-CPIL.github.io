"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a quantile-based heuristic:
    - Ranks candidates by posterior probability of improvement normalized by cost
    - Uses quantile-based scoring to prioritize low-cost regions with high relative improvement likelihood
    - Decouples search from noisy mean estimates
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    start_time = time.time()
    
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Prevent division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Prevent division by zero in cost
    cost_safe = np.maximum(cost, 1e-10)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Fallback to simple cost inversion if running out of time
        scores = 1.0 / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Calculate Z-score for P(f(x) < f_best)
    # Since we are minimizing, improvement is when f(x) is lower than f_best.
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Expected Improvement (for minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Clip to avoid negative values
    ei = np.maximum(ei, 0.0)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 290:
        # Fallback to simple PI/cost if running out of time
        scores = pi / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Quantile-based heuristic:
    # 1. Compute the ratio of probability of improvement to cost
    pi_cost_ratio = pi / cost_safe
    
    # 2. Compute the ratio of expected improvement to cost
    ei_cost_ratio = ei / cost_safe
    
    # 3. Use a blended score that combines both, weighted by uncertainty
    #    When sigma is high, we trust the probability more; when sigma is low, we trust EI more
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Weight for PI: higher when uncertainty is high (explore)
    w_pi = 0.4 * (1.0 - sigma_norm) + 0.3
    # Weight for EI: higher when uncertainty is low (exploit)
    w_ei = 0.4 * sigma_norm + 0.3
    
    # Normalize weights to sum to 1
    w_total = w_pi + w_ei
    w_pi = w_pi / w_total
    w_ei = w_ei / w_total
    
    # Combine the cost-normalized scores
    scores = w_pi * pi_cost_ratio + w_ei * ei_cost_ratio
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 295:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
