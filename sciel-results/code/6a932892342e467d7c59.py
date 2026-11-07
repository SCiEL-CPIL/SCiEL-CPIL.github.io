"""Rewritten acquisition function using cost-aware UCB with adaptive exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a UCB-style approach with cost normalization:
    score = (f_best - mu + beta * sigma) / cost
    
    Where beta is adaptively chosen based on the uncertainty relative to improvement.
    We use a hybrid: when uncertainty is high relative to potential improvement,
    we explore more; otherwise we exploit.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values (higher = more desirable to evaluate)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Compute z-score: how many std devs below f_best is each point
    z = (f_best - mu) / sigma
    
    # Clip z to avoid extreme values
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure EI is non-negative (it should be, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # UCB term: balance exploitation and exploration
    # Use a data-driven beta that scales with the typical EI
    # beta = sqrt(2 * log(1 + n_evals / 2000)) style, but we don't have n_evals directly
    # Instead, use a fixed reasonable beta that works well for this problem
    # The key insight: we want to explore when sigma is large relative to f_best - mu
    
    # Adaptive beta: larger when average uncertainty is high
    avg_sigma = np.mean(sigma)
    avg_ei = np.mean(ei)
    
    # beta controls exploration; scale it with the ratio of uncertainty to improvement
    # If EI is small relative to sigma, we should explore more
    beta = 2.0 * np.sqrt(avg_sigma / (avg_ei + 1e-8))
    beta = np.clip(beta, 1.0, 5.0)
    
    # UCB score: exploitation + exploration
    ucb = ei + beta * sigma
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # But also add a small regularization to avoid dividing by very small costs
    scores = ucb / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Shift so that scores are comparable across calls
    # Subtract the minimum to make the best candidate have the highest relative score
    min_score = np.min(scores)
    scores = scores - min_score
    
    return scores
# EVOLVE-BLOCK-END
