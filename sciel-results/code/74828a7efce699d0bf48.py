"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is non-negative and handle numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For minimization, improvement is f_best - f(x)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Structural Change: Cost-Adjusted UCB
    # The inspiration used pure EI/cost (score 1.0).
    # The current program used a dynamic exploration term (score 0.35).
    # Knowledge indicates that a fixed UCB blend (kappa=1.5) improves performance to 1.06.
    # We adopt the cost-awareness of the inspiration and the exploration benefit
    # of the UCB term, replacing the unstable dynamic scaling of the current program.
    
    kappa = 1.5
    exploration_bonus = kappa * sigma
    
    # Combined Score: Cost-Adjusted (EI + Exploration)
    score = (ei + exploration_bonus) / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
