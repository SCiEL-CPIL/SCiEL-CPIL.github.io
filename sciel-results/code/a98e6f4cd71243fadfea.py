"""Acquisition function for Bayesian optimization of Hartmann-3D using Probability of Improvement (PI) and UCB."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Probability of Improvement (PI) combined with a UCB exploration term.
    
    Structural change: Moves from Expected Improvement (magnitude-focused) to Probability of Improvement (likelihood-focused),
    which is often more effective for multimodal functions with small budgets as it prioritizes finding *any* improvement
    over maximizing the *expected* improvement. The UCB term ensures exploration in high-uncertainty regions.
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    _ensure_start_time()
    
    # Safety guard
    elapsed = time.time() - _START_TIME
    if elapsed > 295:
        # Fallback: Simple UCB
        sigma_safe = np.maximum(sigma, 1e-10)
        scores = -mu + 0.5 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi_score = norm.cdf(z)
    
    # UCB Exploration Term
    # Scale the sigma term to be comparable to the PI score (which is in [0,1])
    # The range of Hartmann-3D is ~7. So sigma is typically < 1.
    # We use a weight to balance the probability (0-1) with the uncertainty.
    # A weight of 0.5 suggests that uncertainty is valued at half the scale of the probability.
    exploration_weight = 0.5
    ucb_component = exploration_weight * sigma_safe
    
    scores = pi_score + ucb_component
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
