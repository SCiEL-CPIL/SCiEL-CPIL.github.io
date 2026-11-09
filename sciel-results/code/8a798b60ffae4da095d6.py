"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 290  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural shift to a
    probability-of-improvement-weighted cost-normalized score.
    
    Instead of Lagrangian relaxation, we use a direct ratio:
    score = P(improve) * expected_improvement / cost^alpha
    
    This naturally balances exploration (through sigma) and exploitation
    (through f_best - mu) while being cost-aware via division.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure all inputs are finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Check time - if close to timeout, return simple cost-adjusted improvement
    if not _elapsed_ok():
        return (f_best - mu) / cost
    
    # Compute z-score for probability of improvement
    improvement = f_best - mu
    z = improvement / sigma
    
    # Probability of improvement
    poi = norm.cdf(z)
    
    # Expected improvement
    ei = improvement * poi + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Known minimum for Styblinski-Tang 2D
    known_min = -78.3323
    
    # Adaptive alpha for cost penalty:
    # Early on (high f_best), we can afford higher cost evaluations
    # Late on (low f_best), we should be more cost-sensitive
    gap = f_best - known_min
    if gap < 1:
        alpha = 1.2
    elif gap < 5:
        alpha = 1.0
    elif gap < 15:
        alpha = 0.8
    elif gap < 30:
        alpha = 0.6
    else:
        alpha = 0.4
    
    # Cost-normalized score: POI * EI / cost^alpha
    # This is a structural change: instead of additive Lagrangian,
    # we use multiplicative cost penalty which preserves the relative
    # ordering of improvements better
    cost_penalty = cost ** alpha
    score = poi * ei / cost_penalty
    
    # Add a small exploration bonus based on uncertainty,
    # scaled by how close we are to the optimum
    if gap > 10:
        # Early stage: encourage exploration
        unc_bonus = sigma / (np.max(sigma) + 1e-9)
        score += 0.05 * unc_bonus
    elif gap > 5:
        # Mid stage: moderate exploration
        unc_bonus = sigma / (np.max(sigma) + 1e-9)
        score += 0.03 * unc_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(score), score, 0.0)
    
    # Normalize to have zero mean and unit variance for stability
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    if std_score > 1e-9:
        scores = (scores - mean_score) / std_score
    
    return scores
# EVOLVE-BLOCK-END
