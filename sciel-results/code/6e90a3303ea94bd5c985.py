"""Acquisition function for Bayesian optimization of Beale 2D using QAF with adaptive target."""
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 330.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Quantile Acquisition Function (QAF) with adaptive target.
    
    QAF maximizes the probability of finding a point better than a target value.
    For minimization, this is P(f(x) < target) = Phi((target - mu) / sigma).
    
    We use an adaptive target that is a fraction below f_best to balance
    exploration (target far below f_best) and exploitation (target close to f_best).
    
    The Beale function has a narrow valley, so we need to balance
    exploration (finding the valley) with exploitation (narrowing in).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive target: how far below f_best to aim
    # When f_best is large, aim further below (more exploration)
    # When f_best is small, aim closer (more exploitation)
    if f_best > 5.0:
        # Early: aim 50% below f_best
        target = f_best * 0.5
    elif f_best > 1.0:
        # Middle: aim 30% below f_best
        target = f_best * 0.7
    elif f_best > 0.1:
        # Late: aim 10% below f_best
        target = f_best * 0.9
    else:
        # Very late: aim 5% below f_best
        target = f_best * 0.95
    
    # Ensure target is below f_best (for minimization, lower is better)
    target = min(target, f_best)
    
    # QAF: P(f(x) < target) = Phi((target - mu) / sigma)
    # Higher probability means higher score
    z = (target - mu) / sigma
    qaf_scores = norm.cdf(z)
    
    # Ensure all scores are finite
    qaf_scores = np.where(np.isfinite(qaf_scores), qaf_scores, 0.0)
    
    # Add tiny noise to break ties
    qaf_scores += 1e-10 * np.random.randn(len(qaf_scores))
    
    return qaf_scores
