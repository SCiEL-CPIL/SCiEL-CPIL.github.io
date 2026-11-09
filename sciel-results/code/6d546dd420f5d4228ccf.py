"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with a UCB term
    and a small random tie-breaker to encourage exploration.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) term
    # Beta controls exploration; use a moderate value
    beta = 2.0
    ucb = mu + beta * sigma_safe
    
    # Combine EI and UCB: use a weighted sum
    # Normalize both to [0, 1] range roughly, or just combine directly
    # Since EI is non-negative and UCB can be any value, we use:
    # score = EI + small_weight * UCB
    # But UCB favors low mu (since we're minimizing), so -UCB would be better for minimization
    # Actually, for minimization, UCB = mu + beta*sigma, and we want to maximize acquisition
    # So we should use -UCB or just EI which already handles this
    
    # Let's use a hybrid: EI + gamma * sigma (extra exploration)
    gamma = 0.5
    scores = ei + gamma * sigma_safe
    
    # Add tiny random noise to break ties and encourage diversity
    # This is crucial: without it, identical points get identical scores
    rng = np.random.default_rng()
    noise = rng.normal(0, 1e-4, size=scores.shape)
    scores = scores + noise
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
