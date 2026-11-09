"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D.
    
    Uses a hybrid Expected Improvement / Upper Confidence Bound approach
    with a softer cost penalty (cost^0.5) to better balance exploration
    and exploitation while avoiding over-penalizing expensive regions.
    
    For the multimodal Rastrigin landscape, we include an adaptive exploration
    term that is stronger early on (when f_best is far from 0) and decays
    as the search converges.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        # Return a simple, fast fallback
        sigma_safe = np.maximum(sigma, 1e-10)
        cost_safe = np.maximum(cost, 1e-10)
        scores = -mu / cost_safe + 0.1 * sigma_safe / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Guard against numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for EI
    improvement = f_best - mu  # positive when mu < f_best
    z = improvement / sigma_safe
    
    # Expected Improvement (positive when mu < f_best)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB term: exploits low mean, explores high uncertainty
    # Beta controls exploration-exploitation tradeoff
    # Use a softer cost penalty: cost^0.5 instead of cost
    beta = 0.5
    ucb = -mu + beta * sigma_safe
    
    # Cost-aware scores using cost^0.5 for softer penalty
    cost_penalty = np.sqrt(cost_safe)
    
    # Hybrid: combine EI and UCB
    # EI is in objective units, UCB is in objective units
    # We blend them with adaptive weights based on f_best
    
    # When f_best is high (early), weight UCB more for exploration
    # When f_best is low (late), weight EI more for exploitation
    if f_best > 10.0:
        w_ei = 0.3
        w_ucb = 0.7
    elif f_best > 5.0:
        w_ei = 0.5
        w_ucb = 0.5
    elif f_best > 2.0:
        w_ei = 0.7
        w_ucb = 0.3
    else:
        w_ei = 0.8
        w_ucb = 0.2
    
    # Combine EI and UCB
    combined = w_ei * ei + w_ucb * ucb
    
    # Apply cost penalty
    scores = combined / cost_penalty
    
    # Add a small exploration bonus for early iterations
    if f_best > 5.0:
        explore_weight = min(1.0, (f_best - 5.0) / 15.0) * 0.05
        explore_term = explore_weight * sigma_safe / cost_penalty
        scores = scores + explore_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
