"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track the start time and use a safety margin.
_DEADLINE = 330.0
_START_TIME = time.time()
_SAFETY_MARGIN = 5.0  # seconds


def _remaining_time():
    """Return remaining time before the deadline, with safety margin."""
    return _DEADLINE - (time.time() - _START_TIME) - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a Cost-Effective Improvement (CEI) approach that estimates
    expected improvement per unit cost, with adaptive exploration.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best observed objective value (lower is better)
    cost : array of shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    score : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Check time budget - if we're running low, return a safe default
    if _remaining_time() < 0.5:
        return np.zeros_like(mu)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost is in range [1, 2] based on the problem description
    # Normalize cost to [0, 1] for better scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-12:
        cost_norm = np.ones_like(cost) * 0.5
    else:
        cost_norm = (cost - cost_min) / cost_range
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) - how much we expect to improve
    from scipy.stats import norm
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (PI) - probability of finding improvement
    pi = norm.cdf(z)
    
    # Cost-Effective Improvement: EI per unit cost
    # This balances the expected gain against the evaluation cost
    cei = ei / (1.0 + cost)
    
    # Exploration term: reward uncertainty where improvement is likely
    # Scale by probability that this region could contain the minimum
    exploration = pi * sigma_safe / (1.0 + cost)
    
    # Adaptive weighting: shift from exploitation to exploration
    # as f_best approaches the known minimum (0 for most functions)
    # Use a soft transition based on how close f_best is to 0
    if f_best > 1e-6:
        # Still far from optimum: balance exploitation and exploration
        # The ratio depends on how much improvement is still possible
        improvement_headroom = f_best  # distance from current best to 0
        exploit_weight = 0.6
        explore_weight = 0.4
    else:
        # Very close to optimum: emphasize exploitation
        exploit_weight = 0.8
        explore_weight = 0.2
    
    # Combine CEI and exploration with adaptive weights
    score = exploit_weight * cei + explore_weight * exploration
    
    # Additional structural change: use a softmax-like sharpening
    # to make the top candidates stand out more clearly
    # This helps the evaluator pick the single best candidate
    # Scale scores to have a meaningful spread
    score_max = np.max(score)
    score_min = np.min(score)
    score_range = score_max - score_min
    
    if score_range > 1e-12:
        # Normalize to [0, 1] then apply a power transformation
        # to emphasize the top candidates
        score_norm = (score - score_min) / score_range
        # Power > 1 makes the distribution more peaked at the top
        score_sharpened = score_norm ** 1.5
    else:
        score_sharpened = score
    
    # Ensure all scores are finite and non-negative
    score_sharpened = np.where(np.isfinite(score_sharpened), score_sharpened, 0.0)
    score_sharpened = np.maximum(score_sharpened, 0.0)
    
    return score_sharpened
# EVOLVE-BLOCK-END
