import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using an Expected Hyperbolic Improvement (EHI) heuristic.
    
    The score is defined as:
    Score = (sigma / cost) * tanh((f_best - mu) / sigma)
    
    This formulation:
    1. Rewards regions where the predicted mean (mu) is better (lower) than the current best (f_best).
    2. Scales the reward by the uncertainty (sigma), encouraging exploration of uncertain promising regions.
    3. Penalizes regions where the predicted mean is worse than the current best (negative score).
    4. Normalizes by cost, making cheap evaluations more attractive.
    5. The tanh function bounds the "improvement" factor, preventing over-exploration of high-uncertainty regions that have poor means (unlike UCB).
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety margin for time (though computation is fast)
    # The problem states 300s limit. NumPy ops on 2000 elements are < 1ms.
    
    # Add small epsilon to sigma to avoid division by zero
    eps = 1e-9
    sigma_safe = sigma + eps
    
    # Calculate normalized improvement z
    # z > 0 means mu < f_best (promising)
    # z < 0 means mu > f_best (less promising)
    z = (f_best - mu) / sigma_safe
    
    # Calculate tanh of z
    # tanh(z) is in [-1, 1]
    # For z >> 0, tanh(z) ~ 1
    # For z << 0, tanh(z) ~ -1
    # For z ~ 0, tanh(z) ~ z
    tanh_z = np.tanh(z)
    
    # EHI Score: sigma * tanh(z) / cost
    # This is bounded by sigma/cost
    scores = (sigma_safe * tanh_z) / cost
    
    # Ensure all scores are finite
    # If cost is 0 (should not happen as cost >= 1), handle it
    # If sigma is inf, handle it
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
