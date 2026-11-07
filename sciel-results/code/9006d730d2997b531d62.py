"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining Probability of Improvement (POI) with
    expected improvement, weighted to favor exploration in the Rosenbrock valley.
    
    Parameters
    ----------
    mu : array of shape (2000,)
        GP mean predictions in objective units.
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed value so far.
    cost : array of shape (2000,)
        Evaluation costs for each candidate.
    
    Returns
    -------
    scores : array of shape (2000,)
        Acquisition values (higher = evaluate next).
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against division by zero and negative costs
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score: how many standard deviations below f_best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (POI)
    poi = norm.cdf(z)
    
    # Combined acquisition: blend EI and POI, scaled by 1/cost
    # Weight POI more heavily to encourage exploration of the valley
    # Use a temperature-like parameter to balance exploitation vs exploration
    # POI is bounded [0,1], EI is unbounded but typically small near optimum
    # Scale EI to be comparable to POI: divide by typical sigma or use raw
    # Here we use a weighted sum: 0.5 * normalized_EI + 0.5 * POI
    # But EI can be negative if mu > f_best (unlikely in practice for minimization)
    # Clip EI to non-negative
    ei_clipped = np.maximum(ei, 0.0)
    
    # Normalize EI by the mean sigma to make it scale-invariant
    mean_sigma = np.mean(sigma_safe)
    if mean_sigma > 1e-9:
        ei_norm = ei_clipped / mean_sigma
    else:
        ei_norm = ei_clipped
    
    # Blend: favor POI for exploration, EI for exploitation
    # The 0.6/0.4 split gives more weight to POI to escape local minima
    combined = 0.6 * poi + 0.4 * ei_norm
    
    # Divide by cost for cost-awareness
    scores = combined / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
