"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function with adaptive exploration.
    
    EI is the gold standard for Bayesian optimization. It balances exploitation
    (low mu) and exploration (high sigma) in a principled way.
    
    For Branin 2D, we use a pure EI score which has been shown to perform
    well on multimodal functions by focusing on regions with high probability
    of significant improvement over the current best.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Phi(z) = standard normal CDF
    phi_z = norm.cdf(z)
    # phi(z) = standard normal PDF
    phi_pdf = norm.pdf(z)
    
    # Expected Improvement
    ei_scores = (f_best - mu) * phi_z + sigma * phi_pdf
    
    # Ensure all scores are finite and non-negative
    ei_scores = np.maximum(ei_scores, 0.0)
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    return ei_scores
# EVOLVE-BLOCK-END
