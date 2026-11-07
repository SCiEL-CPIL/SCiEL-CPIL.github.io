"""Rewritten program: Cost-aware acquisition with adaptive exploration for Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement and UCB.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate, in [1, 2].
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB term: mu + beta * sigma (for minimization, we want low mu + high uncertainty)
    # Adaptive beta: start with moderate exploration, reduce as cost budget is consumed.
    # cost ranges from 1 to 2 per evaluation. Total budget is 30.
    # We use the mean cost as a proxy for how much budget has been used.
    # Early on, cost is low (near 1), so beta should be larger.
    # As cost increases (approaching 2 on average), beta should decrease.
    
    # Use a simple adaptive beta based on the cost distribution
    # beta = base_beta * (1 / (1 + mean_cost / 30))
    # But we don't have total cost, only per-candidate cost.
    # Instead, use the fact that cost = 1 + mean(x_scaled), so cost in [1, 2].
    # Use the median cost as a proxy for current exploration level.
    
    median_cost = np.median(cost)
    # beta starts around 1.0 and decreases as median_cost approaches 2
    # When median_cost = 1 (early), beta = 1.0
    # When median_cost = 2 (late), beta = 0.1
    beta = 1.0 * (2.0 - median_cost) + 0.1
    beta = np.clip(beta, 0.1, 1.0)
    
    # UCB for minimization: we want to minimize, so score = -mu + beta * sigma
    ucb = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-8
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-8
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Combine: weight shifts from exploration (UCB) to exploitation (EI) as cost increases
    # weight_ei increases with median_cost
    weight_ei = np.clip(0.5 + 0.5 * (median_cost - 1.0), 0.3, 0.9)
    weight_ucb = 1.0 - weight_ei
    
    combined = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Divide by cost to make it cost-aware
    cost_safe = np.maximum(cost, 1e-8)
    scores = combined / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
