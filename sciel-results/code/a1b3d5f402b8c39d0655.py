"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are running out of time, use a fast fallback
    if time.time() > _DEADLINE:
        # Simple greedy: minimize mu, penalize cost
        return -mu / (cost + 1e-9)

    epsilon = 1e-9
    
    # 1. Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / (sigma + epsilon)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Use scipy for numerical stability
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware EI: higher is better
    score_ei = ei / (cost + epsilon)

    # 2. Upper Confidence Bound (UCB) for minimization
    # We want to find points with low mu and high sigma (uncertainty)
    # Score = -mu + beta * sigma
    beta = 2.0
    ucb = -mu + beta * sigma
    
    # Cost-aware UCB
    score_ucb = ucb / (cost + epsilon)

    # 3. Blend EI and UCB
    # Equal weighting is a robust starting point
    score = 0.5 * score_ei + 0.5 * score_ucb

    # Ensure output is finite
    score = np.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return score
# EVOLVE-BLOCK-END
