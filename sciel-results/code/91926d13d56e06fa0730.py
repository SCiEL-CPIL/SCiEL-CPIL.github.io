"""Cost-aware acquisition for Hartmann-6d BO loop."""
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = 315.0  # seconds, safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    t_start = time.time()

    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-10)

    # Standardize z-score
    z = (f_best - mu) / sigma_safe

    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Probability of Improvement (POI) as exploration signal
    poi = norm.cdf(z)

    # Cost-aware normalization: divide by cost^beta to prefer cheaper points
    # beta > 1 gives stronger preference for low-cost evaluations
    beta = 1.3
    cost_safe = np.maximum(cost, 1e-6)
    cost_factor = cost_safe ** beta

    # Combine EI (exploitation) and POI (exploration)
    # Higher weight on EI to focus on promising regions
    w_ei = 0.75
    w_poi = 1.0 - w_ei
    base_score = (w_ei * ei + w_poi * poi) / cost_factor

    # Add a UCB-like exploration term scaled by uncertainty
    # This encourages exploring high-variance regions
    ucb_weight = 0.15
    ucb_term = ucb_weight * sigma_safe / cost_factor
    scores = base_score + ucb_term

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline check
    if time.time() - t_start > _DEADLINE:
        return np.zeros_like(mu)

    return scores
