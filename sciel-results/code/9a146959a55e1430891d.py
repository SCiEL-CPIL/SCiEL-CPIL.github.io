"""Rewritten program: Probability of Improvement (PI) acquisition for aggressive exploitation."""
import numpy as np
from scipy.stats import norm
import time

# Track time for safety margin
_t0 = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points using Probability of Improvement.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions in objective units
        sigma: array of shape (2000,) - GP standard deviation predictions in objective units
        f_best: float - lowest observed value so far
    
    Returns:
        scores: array of shape (2000,) - higher means evaluate next
    """
    # Check deadline
    elapsed = time.time() - _t0
    if elapsed > _DEADLINE - 1.0:
        # Return simple regret as fallback
        return np.maximum(f_best - mu, 0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement (PI) for minimization:
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    #
    # This aggressively exploits regions likely to beat the current best,
    # which can rapidly reduce simple regret on multi-modal landscapes.
    
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Compute PI
    scores = norm.cdf(z_clipped)
    
    # Add a very small exploration component to avoid complete exploitation
    # This uses a tiny fraction of the uncertainty to ensure some diversity
    # The weight is kept minimal (0.001) to maintain strong exploitation bias
    exploration_weight = 0.001
    exploration_bonus = exploration_weight * sigma / (np.std(sigma) + 1e-9)
    
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0)
    
    return scores
