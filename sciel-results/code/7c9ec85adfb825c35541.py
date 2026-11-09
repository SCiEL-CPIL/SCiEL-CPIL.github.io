import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0
_total_evals = 0
_ts_rng = None


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4D.
    Uses a hybrid approach: cost-aware EI with adaptive exploration and a
    probability-of-improvement boost for high-confidence improvements.
    """
    global _start_time, _call_count, _total_evals, _ts_rng

    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    _total_evals += 1

    elapsed = time.time() - _start_time
    if elapsed > 315:
        # Safety fallback: simple cost-aware exploitation
        return -mu / np.maximum(cost, 1e-9)

    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-12, 1e6)
    cost_safe = np.maximum(cost, 1e-9)

    # Initialize or reuse the random number generator for Thompson sampling
    if _ts_rng is None:
        _ts_rng = np.random.default_rng(42)

    # Sample a single function value from the GP posterior for each candidate
    # For minimization, we want low sampled values
    z_samples = _ts_rng.standard_normal(size=mu.shape)
    sampled_values = mu - sigma * z_samples

    # Compute improvement per unit cost relative to current best
    # We want to maximize (f_best - sampled_value) / cost
    # This favors points that are likely to improve and have low cost
    improvement = f_best - sampled_values
    scores = improvement / cost_safe

    # Add a small exploration term to encourage visiting uncertain regions
    # This helps escape local optima in the sampled landscape
    exploration_weight = 0.1
    scores += exploration_weight * sigma / cost_safe

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
