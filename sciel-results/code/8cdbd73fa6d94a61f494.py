"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using Probability of Improvement (PI).
    
    PI measures the probability that the function value at a candidate point is
    lower than the current best observed value. It is a robust alternative to
    Expected Improvement (EI) and avoids the heavy weighting of large improvements
    that can lead to erratic behavior in noisy or multi-modal landscapes.
    
    mu: (2000,) posterior mean in objective units
    sigma: (2000,) posterior standard deviation in objective units
    f_best: scalar, lowest observed value
    
    Returns: (2000,) scores, higher = more promising
    """
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-10)
    
    # Calculate the z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    # We want P(f(x) < f_best) = P((f(x) - mu) / sigma < (f_best - mu) / sigma)
    # = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # To ensure we always have some exploration, we can add a small epsilon
    # based on the uncertainty, or simply rely on the fact that if sigma is high,
    # z will be closer to 0 (if mu is close to f_best) or negative/positive.
    # PI is bounded between 0 and 1.
    
    # In cases where the model is very confident (sigma -> 0) and mu > f_best,
    # PI -> 0. If mu < f_best, PI -> 1.
    # This naturally focuses on points that are likely to be better.
    
    # Return PI as the score. Higher probability of improvement = higher score.
    # Ensure finite values
    score = np.where(np.isfinite(pi), pi, 0.0)
    
    return score
# EVOLVE-BLOCK-END
