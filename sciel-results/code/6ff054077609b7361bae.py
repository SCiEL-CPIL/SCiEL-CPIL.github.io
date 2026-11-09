"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a pure Upper Confidence Bound (UCB) strategy.
    In the low-budget regime (30 evaluations), standard EI can be too greedy 
    and get stuck in local minima. UCB with a sufficient beta encourages 
    exploration of regions with high posterior uncertainty, which is crucial 
    for escaping local optima in the Ackley function.
    
    mu: (N,) posterior mean
    sigma: (N,) posterior std
    f_best: best observed value so far
    
    Returns: (N,) scores, higher is better.
    """
    # Ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # UCB: -mu + beta * sigma
    # Higher beta encourages more exploration.
    # Based on findings, a value around 1.0-1.5 often outperforms EI hybrids 
    # in early stages of low-budget BO for multimodal functions.
    beta = 1.5
    
    score = -mu + beta * sigma_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
