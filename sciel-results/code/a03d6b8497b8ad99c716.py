import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    cost = np.maximum(np.asarray(cost, dtype=np.float64), 1e-8)
    
    # Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure ei is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: multiply EI by exponential penalty on cost
    # This suppresses high-cost regions more aggressively than linear division
    # Cost ranges from 1.0 (corners) to 2.0 (center), so exp(-cost) gives
    # exp(-2.0)~0.135 at center and exp(-1.0)~0.368 at corners
    # We normalize to make the penalty relative: lower cost = higher multiplier
    # Using exp(-(cost - 1.0)) so center (cost=2) gets exp(-1)~0.37, 
    # corners (cost=1) get exp(0)=1.0
    cost_penalty = np.exp(-(cost - 1.0))
    
    score = ei * cost_penalty
    
    # Small exploration bonus to avoid premature convergence
    exploration = 0.001 * sigma * cost_penalty
    
    return score + exploration
