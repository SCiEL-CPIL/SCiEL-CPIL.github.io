"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining cost-adjusted EI with an
    exploration bonus that favors low-cost regions early in the search.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns
    -------
    ndarray, shape (2000,)
        Acquisition scores (higher = prefer to evaluate next).
    """
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-scores for expected improvement
    z = (f_best - mu) / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Standard expected improvement
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Cost-adjusted EI: divide by cost to penalize expensive evaluations
    # This encourages the algorithm to explore cheaper regions
    ei_cost = ei / cost
    
    # Exploration bonus: add a term that rewards uncertainty in cheap regions
    # The beta parameter controls exploration vs exploitation trade-off
    # Use a dynamic beta that decreases as we get more information
    # (approximated by how close f_best is to 0, since min of levy2d is 0)
    # When f_best is large, we're early in search -> explore more
    # When f_best is small, we're late -> exploit more
    
    # Scale factor based on how far we are from the optimum
    # f_best ranges from ~100+ (levy2d max) down to 0
    # Use a smooth transition
    exploration_scale = np.exp(-0.1 * np.abs(f_best))
    
    # Exploration term: sigma / cost, weighted by exploration_scale
    # This encourages exploration in low-cost regions
    exploration = exploration_scale * sigma / cost
    
    # Combine: primarily cost-adjusted EI with a smaller exploration component
    # The exploration component helps escape local optima early on
    score = ei_cost + 0.3 * exploration
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
