"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 315  # 315s limit, 15s margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
    
    # Cost-aware acquisition function
    # Combines Expected Improvement (EI) for exploitation with UCB for exploration
    # Normalized by cost to prefer cheaper evaluations
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal
    from scipy.stats import norm
    
    z = (f_best - mu) / (sigma + 1e-9)
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # UCB term for exploration
    beta = 2.0
    ucb = beta * sigma
    
    # Combine EI and UCB with a weight
    # Use a weighted combination that balances exploitation and exploration
    weight_ei = 1.0
    weight_ucb = 0.5
    
    acquisition_score = weight_ei * ei + weight_ucb * ucb
    
    # Normalize by cost (cost is 1 + mean(x scaled to [0,1]), so range [1, 2])
    # Divide by cost to prefer cheaper evaluations
    score = acquisition_score / (cost + 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
