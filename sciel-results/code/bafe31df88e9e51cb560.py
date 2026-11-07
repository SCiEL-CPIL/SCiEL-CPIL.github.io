"""Rewritten program using UCB acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Parameters:
    mu: Array of mean predictions (2000,).
    sigma: Array of standard deviation predictions (2000,).
    f_best: Best observed function value so far.
    
    Returns:
    scores: Array of acquisition scores (2000,). Higher scores indicate better candidates.
    """
    # UCB for minimization: We want to maximize the lower bound of the function value.
    # Lower bound L(x) = mu(x) - kappa * sigma(x).
    # We want to select x that minimizes L(x).
    # Since the evaluator selects the candidate with the HIGHEST score,
    # we return -L(x) = -mu(x) + kappa * sigma(x).
    
    # A fixed kappa of 2.5 provides a strong balance between exploration and exploitation
    # suitable for the small budget (25 BO steps) and multimodal nature of Styblinski-Tang.
    kappa = 2.5
    
    scores = -mu + kappa * sigma
    
    # Ensure finite scores
    # sigma and mu should be finite given the GP setup, but just in case:
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
