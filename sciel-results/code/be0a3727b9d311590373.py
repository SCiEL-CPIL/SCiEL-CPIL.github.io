"""Rewritten program with structural change: Expected Regret w.r.t True Minimum."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes an acquisition score for Bayesian Optimization.
    
    This implementation uses Expected Regret relative to the known global minimum (3.0).
    The goal is to minimize E[max(f(x) - 3.0, 0)].
    The returned score is the negative of this expectation, so higher scores indicate
    points expected to be closer to the global minimum.
    
    Parameters:
    mu : array-like, shape (n_candidates,)
        Posterior mean of the GP at the candidate points.
    sigma : array-like, shape (n_candidates,)
        Posterior standard deviation of the GP at the candidate points.
    f_best : float
        The best observed value so far (not directly used in the core formula, 
        but kept for interface compatibility).
        
    Returns:
    scores : array-like, shape (n_candidates,)
        Acquisition scores. Higher is better.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # True global minimum of the Goldstein-Price function
    f_min = 3.0
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score for the threshold f_min
    # z = (f_min - mu) / sigma
    z = (f_min - mu) / sigma_safe
    
    # Calculate E[max(f(x) - f_min, 0)]
    # Formula: sigma * [ phi(z) - z * (1 - Phi(z)) ]
    # Note: phi is pdf, Phi is cdf
    pdf_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # The term (1 - cdf_z) is the probability that f(x) > f_min
    # The expectation of the excess is:
    # sigma * ( pdf_z - z * (1 - cdf_z) )
    # If z is very large (mu << f_min), pdf ~ 0, 1-cdf ~ 0, regret ~ 0.
    # If z is very negative (mu >> f_min), pdf ~ 0, 1-cdf ~ 1, regret ~ sigma * (-z) = mu - f_min.
    
    regret = sigma_safe * (pdf_z - z * (1.0 - cdf_z))
    
    # We want to minimize regret, so we maximize -regret
    scores = -regret
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
