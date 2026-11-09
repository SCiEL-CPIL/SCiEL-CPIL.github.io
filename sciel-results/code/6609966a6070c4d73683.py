"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track elapsed time to ensure we finish well within the 300s limit
_START_TIME = time.time()
DEADLINE_LIMIT = 300.0
SAFETY_MARGIN = 10.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining_budget():
    return DEADLINE_LIMIT - _elapsed() - SAFETY_MARGIN

def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """
    Standard Expected Improvement for minimization.
    EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu - xi) / sigma
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _upper_confidence_bound(mu, sigma, beta):
    """
    Upper Confidence Bound for minimization: mu - beta * sigma
    """
    return mu - beta * sigma

def _probability_of_improvement(mu, sigma, f_best):
    """
    Probability of Improvement for minimization: P(f(x) < f_best).
    PI = Phi((f_best - mu) / sigma)
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)

def _cost_adjusted_ei(mu, sigma, f_best, cost, xi=0.01):
    """
    Cost-adjusted Expected Improvement: EI / cost
    """
    ei = _expected_improvement(mu, sigma, f_best, xi=xi)
    cost_safe = np.maximum(cost, 1e-12)
    return ei / cost_safe

def _ucb_cost_aware(mu, sigma, f_best, cost, beta):
    """
    Cost-aware UCB: (f_best - UCB) / cost
    """
    ucb = _upper_confidence_bound(mu, sigma, beta)
    improvement = np.maximum(f_best - ucb, 0.0)
    cost_safe = np.maximum(cost, 1e-12)
    return improvement / cost_safe

def _quantile_improvement(mu, sigma, f_best, alpha=0.1):
    """
    Expected reduction in the alpha-quantile of f(x).
    For a Gaussian posterior N(mu, sigma^2), the alpha-quantile is mu + sigma * z_alpha.
    The expected reduction in this quantile relative to f_best is:
    max(0, f_best - (mu + sigma * z_alpha))
    where z_alpha is the standard normal quantile (negative for alpha < 0.5).
    """
    z_alpha = norm.ppf(alpha)
    quantile = mu + sigma * z_alpha
    improvement = np.maximum(f_best - quantile, 0.0)
    return improvement

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Quantile-Based acquisition.
    Maximizes the expected reduction in the posterior lower quantile (10th percentile)
    normalized by evaluation cost.
    
    Args:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Quantile-based acquisition with alpha=0.1 (10th percentile)
    # This targets robust improvement by focusing on the lower tail of the posterior
    alpha = 0.1
    qi = _quantile_improvement(mu, sigma, f_best, alpha=alpha)
    
    # Cost-adjusted: divide by cost to prioritize cheap evaluations
    score = qi / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
