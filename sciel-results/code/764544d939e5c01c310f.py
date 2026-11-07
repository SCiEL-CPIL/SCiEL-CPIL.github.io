"""Cost-aware acquisition for Powell 4D BO using Probability of Improvement.

Structural change: replace additive UCB with multiplicative PI-based scoring.
PI = Phi((f_best - mu) / sigma) naturally saturates at 1.0 for certain improvements
and decays smoothly with uncertainty. Apply soft cost penalty via exponentiation
(PI^(1/cost)) to prioritize cheap, high-confidence improvements.
"""
import numpy as np
from scipy.stats import norm
import time

# ------------------------------- deadline guard ---------------------------------
_T0 = time.time()
_DEADLINE = 330.0
_MARGIN = 15.0


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _T0)


# ------------------------------- public entry point ------------------------------
def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement acquisition function.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Lowest observed function value
    cost : array of shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x_scaled_to_01)
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1.0)

    # Probability of Improvement: PI = Phi((f_best - mu) / sigma)
    # This is the probability that f(x) < f_best under the GP posterior
    z = (f_best - mu) / sigma
    pi = norm.cdf(z)  # Values in [0, 1], higher = more likely to improve

    # Clip to valid range for exponentiation
    pi = np.clip(pi, 1e-12, 1.0 - 1e-12)

    # Cost-aware scoring: PI^(1/cost)
    # Higher cost reduces the score, prioritizing cheaper evaluations
    # This is multiplicative in log-space: log(score) = log(PI) / cost
    # Since log(PI) <= 0, dividing by larger cost makes it less negative (closer to 0)
    # But we want higher cost to give LOWER score, so we use:
    # score = PI * exp(-alpha * (cost - 1)) for a soft penalty
    # Or more simply: score = PI / (cost^beta) to penalize expensive points
    
    # Using a soft power penalty: score = PI * (1 / cost^0.5)
    # This gives a gentle preference for cheaper points while maintaining PI's structure
    alpha = 0.5  # Cost sensitivity parameter
    score = pi * np.power(cost, -alpha)

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
