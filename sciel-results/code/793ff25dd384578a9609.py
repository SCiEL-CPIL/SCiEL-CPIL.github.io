"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track start time for deadline guard
_START_TIME = time.time()
DEADLINE = _START_TIME + 315.0  # 15s safety margin before 330s kill


def _expected_improvement(mu, sigma, f_best):
    """Compute Expected Improvement (EI) for minimization."""
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    return ei


def _ucb(mu, sigma, beta):
    """Compute UCB for minimization (lower mu is better)."""
    return -mu + beta * sigma


def _pi(mu, sigma, f_best):
    """Probability of Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI, UCB, and PI.
    
    Strategy: Use a weighted combination of EI and UCB, normalized by cost.
    EI handles exploitation, UCB handles exploration.
    """
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > DEADLINE - 10:
        # Return a safe default (all zeros) if we're running out of time
        return np.zeros(len(mu))
    
    n = len(mu)
    
    # Compute individual acquisition components
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Adaptive beta for UCB: increase exploration early, decrease later
    # Use the magnitude of f_best as a proxy for how much we've improved
    # If f_best is close to 0, we're near optimum, so reduce exploration
    f_best_abs = abs(f_best)
    # beta ranges from ~1.0 (high exploration) to ~0.1 (low exploration)
    beta = 0.1 + 0.9 * min(1.0, f_best_abs)
    
    ucb = _ucb(mu, sigma, beta)
    
    # Normalize EI and UCB to [0, 1] range for fair combination
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min if ucb_range > 1e-12 else 1.0
    ucb_norm = (ucb - ucb_min) / ucb_range
    ei_norm = ei / ei_max
    
    # Weighted combination: 60% EI, 40% UCB
    combined = 0.6 * ei_norm + 0.4 * ucb_norm
    
    # Cost-aware: divide by cost (higher cost = lower score)
    cost_safe = np.maximum(cost, 1e-9)
    scores = combined / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
