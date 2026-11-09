"""Acquisition function for Bayesian optimization of a 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """Upper Confidence Bound acquisition for minimization.
    
    Args:
        mu: array of shape (2000,) with posterior means
        sigma: array of shape (2000,) with posterior std devs
        f_best: best observed value so far
        
    Returns:
        array of shape (2000,) with acquisition scores (higher = evaluate next)
    """
    # UCB for minimization: prefer low mu and high uncertainty
    beta = 2.0
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
