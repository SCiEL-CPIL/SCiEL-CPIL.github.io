"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive UCB.
    
    Strategy:
    Instead of Expected Improvement (EI) which is myopic and can be overly
    exploitative in early stages or sensitive to the exact value of f_best,
    we use an Upper Confidence Bound (UCB) for minimization (i.e., Lower
    Confidence Bound in value, but we maximize the score).
    
    Score(x) = (-mu(x) + beta * sigma(x)) / cost(x)
    
    The parameter beta is adapted based on the current minimum uncertainty
    observed in the candidate set. If the GP is very confident (low sigma),
    we reduce exploration (lower beta). If uncertainty is high, we increase
    exploration (higher beta). This helps balance exploration and exploitation
    without knowing the exact iteration count.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero or negative costs (cost is always >= 1 here)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate the minimum sigma to gauge the level of uncertainty
    # If min sigma is very small, the GP is confident in some region -> exploit more
    # If min sigma is large, the GP is uncertain -> explore more
    min_sigma = np.min(sigma)
    
    # Adaptive beta:
    # Base beta of 1.0.
    # Add up to 2.0 extra exploration if min_sigma is large.
    # Decay rate: if min_sigma > 0.5, beta is near 3.0.
    # If min_sigma < 0.1, beta is near 1.0.
    beta = 1.0 + 2.0 * np.exp(-min_sigma / 0.2)
    
    # UCB for minimization: we want low mu and high sigma (uncertainty)
    # We maximize the score, so we negate mu.
    # ucb_value = -mu + beta * sigma
    ucb_value = -mu + beta * sigma
    
    # Cost-aware: divide by cost
    scores = ucb_value / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
