"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with a small
    exploration bonus proportional to sigma.
    
    EI encourages exploitation of promising regions, while the sigma bonus
    encourages exploration of uncertain regions, helping escape local optima.
    """
    # Compute Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add small exploration bonus to encourage exploring uncertain regions
    # This helps escape local optima in multimodal functions like Hartmann 3D
    exploration_bonus = 0.1 * sigma
    
    return ei + exploration_bonus
# EVOLVE-BLOCK-END
