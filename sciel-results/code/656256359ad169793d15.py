"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is non-negative and has a minimum floor to avoid division by zero
    # sigma is already a standard deviation, so it should be >= 0
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We use scipy.stats.norm for Phi (cdf) and phi (pdf)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (handling numerical precision issues)
    ei = np.maximum(ei, 0.0)
    
    # Exploration term: P(improve) * sigma
    # P(improve) is the probability that f(x) < f_best, which is Phi(z)
    exploration = cdf_z * sigma
    
    # Total benefit
    benefit = ei + exploration
    
    # Cost-aware penalty: sublinear 1/sqrt(cost)
    # cost is in [1, 2]
    # Using sqrt to soften the penalty compared to 1/cost
    cost_penalty = 1.0 / np.sqrt(cost)
    
    # Final score
    score = benefit * cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
