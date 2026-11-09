"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
DEADLINE = _START_TIME + 315.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Griewank 2D.
    
    Structural change: Uses a penalized Expected Improvement (EI) where the
    improvement term is scaled by exp(-alpha * sigma). This explicitly
    discounts the expected improvement in regions of high uncertainty,
    forcing the search to converge on confident improvements rather than
    exploring wide areas of the space. This is combined with a cost-aware
    normalization that is adaptive to the current state of the search.
    """
    # Check deadline - if we're close to timeout, return a safe fallback
    if time.time() > DEADLINE:
        # Fallback: use a simple PI / cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        scores = pi / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(scores), scores, 0.0)

    n = len(mu)
    
    # Compute Expected Improvement for minimization
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Structural change: Penalize EI by uncertainty
    # exp(-alpha * sigma) reduces the weight of EI in high-uncertainty regions
    # This encourages the algorithm to exploit known good regions once they are found
    # and avoids wasting evaluations on highly uncertain areas
    alpha = 5.0
    uncertainty_penalty = np.exp(-alpha * sigma_safe)
    
    # Combine EI with the penalty
    penalized_ei = ei * uncertainty_penalty
    
    # Add a small exploration term to avoid completely ignoring uncertain regions
    # This term is scaled by sigma to ensure some exploration happens
    # but it's much smaller than the main EI term
    exploration_term = 0.1 * sigma_safe
    
    # Combine penalized EI and exploration
    combined = penalized_ei + exploration_term
    
    # Cost-aware normalization
    # The cost is 1 + mean(x_scaled), so it's between 1 and 2
    # We divide by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-9)
    scores = combined / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
