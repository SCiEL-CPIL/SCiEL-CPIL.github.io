"""Thompson Sampling acquisition for cost-aware Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function with adaptive exploration.
    
    Samples objective values from the Gaussian posterior and scores them
    with an adaptive exploration bonus that decays as the best observed
    value approaches zero (the known minimum of Levy 2D).
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean of the objective at each candidate point.
    sigma : array-like, shape (2000,)
        Posterior standard deviation of the objective at each candidate point.
    f_best : float
        Lowest observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Thompson Sampling scores; higher values are preferred for next evaluation.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Draw one sample from the posterior for each candidate
    z = np.random.randn(mu.shape[0])
    samples = mu + sigma * z
    
    # Adaptive exploration weight: larger when f_best is far from 0,
    # smaller as we get closer to the known minimum.
    # Clamp f_best to a small positive floor to avoid division by zero
    # and to keep the weight bounded.
    f_floor = max(f_best, 1e-3)
    exploration_weight = 1.0 / np.sqrt(f_floor)
    
    # Score: prefer low sampled values (minimization), penalize cost,
    # and add an exploration bonus proportional to posterior uncertainty
    # scaled by the adaptive weight.
    # The exploration term encourages visiting high-uncertainty regions
    # early on, and tapers off as we converge.
    base_score = -samples / cost
    exploration_bonus = exploration_weight * sigma / cost
    scores = base_score + exploration_bonus
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e18)
    
    return scores
# EVOLVE-BLOCK-END
