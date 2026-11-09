# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s kill


def _check_deadline():
    """Return True if we are close to the deadline."""
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a hybrid approach combining:
    1. Expected Improvement (EI) for exploitation
    2. Uncertainty (sigma) for exploration
    3. A deadline guard to ensure we return before timeout
    
    The Weierstrass function on [-0.5, 0.5] has its global minimum near 0.
    We use a weighted combination of EI and UCB to balance exploitation and exploration.
    """
    # Deadline check: if we're too close to the limit, return a simple score
    if _check_deadline():
        # Fallback: pure EI with a small exploration bonus
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei + 0.05 * sigma_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) for minimization
    # Higher beta means more exploration
    # We use a slightly higher beta than typical to encourage exploration
    # since the Weierstrass function is highly oscillatory
    beta = 1.5
    ucb = mu - beta * sigma_safe
    
    # Combine EI and UCB with a dynamic weight
    # EI is in the same units as the objective, UCB is also in objective units
    # We normalize them to make the combination more meaningful
    
    # Scale EI to be comparable to UCB
    # EI is always non-negative, UCB can be negative (which is good)
    # We want higher scores to mean "evaluate next"
    
    # Simple combination: weighted sum
    # EI rewards points where we expect improvement over f_best
    # UCB rewards points with low mu and high sigma
    w_ei = 0.5
    w_ucb = 0.5
    
    # For UCB, lower values are better, so we negate it for the score
    score = w_ei * ei + w_ucb * (-ucb)
    
    # Add a small bonus for uncertainty to encourage exploration
    score += 0.05 * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
