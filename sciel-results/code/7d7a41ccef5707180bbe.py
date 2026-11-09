"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _norm_cdf_vec(z):
    """Vectorized standard normal CDF with numerical stability."""
    return 0.5 * (1.0 + np.vectorize(math_erf)(z / np.sqrt(2.0)))


def _norm_pdf_vec(z):
    """Vectorized standard normal PDF."""
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def _math_erf(x):
    """Fallback erf using scipy if needed."""
    return float(norm.cdf(x * np.sqrt(2.0)) * 2.0 - 1.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP surrogate at the 2000 candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP surrogate at the 2000 candidate points.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate (higher = evaluate next).
    """
    t_start = time.time()
    timeout_limit = 280.0
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # z-score: how many std devs below f_best the mean is
    z = (f_best - mu) / sigma_safe
    
    # Probability that f(x) < f_best (probability of improvement)
    p_improve = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * p_improve + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Structural change: multiplicative combination of probability-weighted EI
    # and a cost-efficiency factor, rather than pure division.
    # 
    # The key insight: dividing EI by cost can over-exploit cheap points that
    # have low actual improvement potential. Instead, we use:
    #   score = (p_improve * EI) / (cost^alpha)
    # where alpha < 1 softens the cost penalty, and the p_improve weighting
    # ensures we prioritize points with high probability of actual improvement.
    #
    # Additionally, we add a knowledge-gradient-inspired term that rewards
    # points where the posterior is most informative about the minimum.
    
    # Soft cost penalty with alpha < 1
    alpha = 0.6
    cost_factor = cost_safe ** alpha
    
    # Primary term: probability-weighted EI with soft cost
    primary = (p_improve * ei) / cost_factor
    
    # Knowledge-gradient-inspired exploration: reward high uncertainty
    # in regions where the mean is not too far above f_best
    # This captures the value of information
    kg_exploration = sigma_safe * np.exp(-0.5 * np.maximum(z, 0.0) ** 2) / cost_factor
    
    # Combine: primary exploitation + knowledge-gradient exploration
    # Weight to balance exploitation vs exploration
    w_exploit = 1.0
    w_explore = 0.15
    
    scores = w_exploit * primary + w_explore * kg_exploration
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    if time.time() - t_start > timeout_limit:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
