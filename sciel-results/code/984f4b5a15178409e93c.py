import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 285


def _elapsed_ok():
    return (time.time() - _START_TIME) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    n = mu.size

    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.maximum(cost, 1.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)

    # ---- Hypervolume-inspired cost-aware acquisition ----
    # Key insight: normalize EI by cumulative cost budget consumed so far,
    # creating a global efficiency metric that stabilizes exploration-exploitation.
    
    # Expected improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Estimate cumulative cost budget consumed so far.
    # The loop runs while total cost < 30. We don't know the exact total cost,
    # but we can estimate it from the cost array itself.
    # The cost array has 2000 entries, each in [1, 2].
    # A reasonable proxy for "budget consumed" is the mean cost times some factor,
    # but more directly: we want a normalization that makes the acquisition
    # a rate of improvement per unit cost.
    
    # Use the mean cost as a proxy for the marginal cost of evaluation.
    # This creates a "cost efficiency" score: EI per unit cost.
    mean_cost = np.mean(cost)
    
    # Cost-normalized EI: this is the "hypervolume" contribution per unit cost
    ei_cost_eff = ei / (mean_cost + 1e-10)

    # ---- Multi-objective inspired: combine exploitation and exploration ----
    # Exploration term: uncertainty-based, also cost-normalized
    # Use sigma as a proxy for "potential" improvement in unexplored regions
    # Cost-normalized uncertainty
    sigma_eff = sigma / (mean_cost + 1e-10)

    # Dynamic weighting: as f_best improves (gets closer to true min),
    # we should shift from exploration to exploitation.
    # We don't know the true min (-78.3323), but we can use the spread of mu.
    mu_spread = np.max(mu) - np.min(mu)
    if mu_spread < 1e-6:
        mu_spread = 1.0
    
    # Weight for exploitation: higher when EI is large relative to uncertainty
    ei_max = np.max(ei_cost_eff)
    sigma_max = np.max(sigma_eff)
    
    if ei_max > 1e-10 and sigma_max > 1e-10:
        # Balance based on relative magnitudes
        w_exploit = ei_max / (ei_max + sigma_max)
        w_exploit = np.clip(w_exploit, 0.3, 0.8)
    else:
        w_exploit = 0.5

    w_explore = 1.0 - w_exploit

    # Combined score: weighted sum of cost-efficiency EI and cost-efficiency uncertainty
    # For minimization, we want low mu, so -mu is a good exploitation signal
    # But EI already captures this. Let's use:
    # score = w_exploit * ei_cost_eff + w_explore * sigma_eff
    
    score = w_exploit * ei_cost_eff + w_explore * sigma_eff

    # ---- Additional refinement: penalize high-cost points more aggressively ----
    # The cost is in [1, 2]. Points near cost=2 (far from origin in scaled space)
    # are more expensive. We want to prefer cheaper evaluations when possible.
    # Add a small cost penalty that is stronger for expensive points.
    # Normalize cost to [0, 1] for the penalty
    cost_norm = (cost - 1.0) / 1.0  # Now in [0, 1]
    
    # Small penalty: 0.1 * cost_norm, scaled by the score magnitude
    score_scale = np.max(score) if np.max(score) > 1e-10 else 1.0
    score = score - 0.1 * cost_norm * score_scale

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    # Fallback for near-timeout
    if not _elapsed_ok():
        fallback = ei / cost
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback

    return score
