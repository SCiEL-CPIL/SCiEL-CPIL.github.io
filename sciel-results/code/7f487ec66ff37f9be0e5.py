"""Rewritten acquisition function for cost-aware BO on Rastrigin 2D."""
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Pareto-frontier inspired score combining
    expected improvement per unit cost with a robust exploration term.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative costs and sigma
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # EI per unit cost
    ei_per_cost = ei / cost
    
    # Exploration term: normalized sigma per unit cost
    # Use a robust scaling: divide by median sigma to normalize
    sigma_med = np.median(sigma)
    if sigma_med < 1e-9:
        sigma_med = 1.0
    sigma_norm = sigma / sigma_med
    explore_per_cost = sigma_norm / cost
    
    # Combine: use a weighted sum that balances exploitation and exploration
    # The weight on exploration adapts based on how much improvement is possible
    # When EI is large relative to sigma, exploit more; otherwise explore more
    ei_med = np.median(ei)
    if ei_med < 1e-9:
        ei_med = 1.0
    ei_norm = ei / ei_med
    
    # Adaptive weight: more exploration when EI is small relative to uncertainty
    # weight_explore = 1.0 when ei_norm is small, 0.0 when ei_norm is large
    weight_explore = np.clip(1.0 - ei_norm, 0.0, 1.0)
    weight_exploit = 1.0 - weight_explore
    
    # Combined score
    scores = weight_exploit * ei_per_cost + weight_explore * explore_per_cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
