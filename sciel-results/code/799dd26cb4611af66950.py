"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement with UCB exploration bonus.
    
    Uses proper Expected Improvement (EI) for minimization, normalized by
    evaluation cost, plus a small UCB term for exploration in early stages.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # UCB exploration bonus: add a small multiple of sigma to encourage
    # exploration in high-uncertainty regions. This helps in early iterations
    # when the GP is still learning the landscape.
    # Use a decaying beta: start with moderate exploration, reduce over time
    # We can't track iteration count directly, but we can use the magnitude
    # of sigma relative to the range of mu as a proxy for "how uncertain" we are.
    # Simpler: use a fixed small beta.
    beta = 0.3
    ucb_bonus = beta * sigma_safe
    
    # Combined score before cost normalization
    raw_score = ei + ucb_bonus
    
    # Cost normalization: divide by cost to get "improvement per unit cost"
    # This is the standard approach for cost-aware BO
    score = raw_score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
