"""Acquisition function for Bayesian optimization of hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition with a small improvement threshold.
    
    EI is a principled acquisition function that balances exploration and exploitation
    naturally. For minimization, EI(x) = E[max(f_best - f(x), 0)].
    
    We use a small threshold xi to avoid exploiting already-found good points too much,
    which encourages mild exploration. The standard formula is:
    
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    
    With a threshold:
    EI(x) = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - xi - mu) / sigma
    
    This is the classic EI formulation. We use xi=0.01 as a small improvement
    threshold to balance exploitation and exploration.
    """
    # Small improvement threshold to encourage mild exploration
    xi = 0.01
    
    # Compute z-score
    z = (f_best - xi - mu) / (sigma + 1e-12)
    
    # Compute EI: (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite values
    score = np.where(np.isfinite(ei), ei, 0.0)
    
    return score
# EVOLVE-BLOCK-END
