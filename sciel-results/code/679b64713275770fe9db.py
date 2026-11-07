"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for safety
_start_time = time.time()
_time_limit = 300  # Seconds


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending Expected Improvement (EI) and UCB.
    
    EI provides proper Bayesian calibration of improvement vs. uncertainty,
    avoiding UCB's tendency to over-explore in low-noise regimes. A quadratic
    cost penalty more sharply discriminates between cheap and expensive
    evaluations under tight budget. The EI/UCB blend maintains exploration
    pressure without the saturation issues of pure UCB.
    
    The improvement is normalized by cost so that cheaper evaluations receive
    proportionally higher credit for the same absolute improvement.
    """
    # Time guard: if we're getting close to the limit, return a safe default
    elapsed = time.time() - _start_time
    if elapsed > _time_limit - 5:  # 5 second safety margin
        return np.zeros_like(mu)
    
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    imp = f_best - mu
    z = imp / sigma
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Expected Improvement
    ei = imp * Phi + sigma * phi
    
    # UCB component for exploration
    # A moderate beta to balance exploration and exploitation
    beta = 1.0
    ucb = imp + beta * sigma
    
    # Blend EI and UCB: EI provides calibrated improvement, UCB adds exploration
    # Use 0.7 * EI + 0.3 * UCB to favor calibrated improvement while maintaining exploration
    blended = 0.7 * ei + 0.3 * ucb
    
    # Quadratic cost penalty: more sharply discriminates between cheap and expensive evaluations
    # Cost is in [1, 2], so cost^2 is in [1, 4], giving up to 4x penalty for expensive points
    score = blended / (cost ** 2)
    
    # Ensure output is finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
