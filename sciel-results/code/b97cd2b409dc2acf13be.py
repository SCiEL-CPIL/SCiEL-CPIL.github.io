"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP in objective units.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP in objective units.
    f_best : float
        Best observed objective value so far (minimum, since we minimize).
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point, in [1, 2].
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Structural Change: Switch from Expected Improvement (EI) to Cost-Aware UCB.
    # UCB encourages exploration of high-uncertainty regions, which can help escape
    # local minima in multimodal functions like Rastrigin and Ackley.
    # The cost-aware division ensures we prefer cheaper evaluations when the
    # upper confidence bound is similar.
    
    # UCB = mu + kappa * sigma
    # For minimization, we want to maximize (f_best - UCB) or minimize UCB?
    # Standard UCB for minimization: Score = - (mu - kappa * sigma) = -mu + kappa * sigma
    # Or more commonly, we maximize the "regret" potential.
    # Let's use the standard form for maximization of improvement potential:
    # We want to find x that minimizes f(x).
    # The "value" of a point is its predicted minimum potential.
    # UCB for minimization: f(x) ~ N(mu, sigma).
    # We want to maximize the probability that f(x) < f_best.
    # A common heuristic is to maximize: -mu + kappa * sigma.
    # This favors low mu (good prediction) and high sigma (uncertainty).
    
    kappa = 2.0  # Exploration parameter
    
    # Calculate UCB score (higher is better)
    # Note: We are minimizing, so lower mu is better.
    # Score = -mu + kappa * sigma
    ucb_score = -mu + kappa * sigma_safe
    
    # Cost-aware score: divide by cost
    # Higher UCB score and lower cost = higher acquisition score
    score = ucb_score / np.maximum(cost, 1e-8)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
