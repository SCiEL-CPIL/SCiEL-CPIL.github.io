"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    deadline = 300.0  # seconds
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Prevent division by zero and negative sigma
    sigma_safe = np.where(sigma < 1e-9, 1e-9, sigma)
    
    # 1. Calculate Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Calculate Upper Confidence Bound (UCB) for minimization
    # We want high scores for low mu and high sigma.
    # UCB_min = -mu + beta * sigma
    # We choose beta adaptively based on the spread of sigma to ensure exploration is meaningful
    # A fixed beta of 2.0 is a common heuristic, but scaling by std(sigma) can be more robust
    beta = 2.0 * np.std(sigma_safe) if np.std(sigma_safe) > 1e-9 else 2.0
    ucb = -mu + beta * sigma_safe
    
    # 3. Normalize scores to [0, 1] range to make them comparable and robust to scale
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-9:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    ei_norm = normalize_scores(ei)
    ucb_norm = normalize_scores(ucb)
    
    # 4. Combine EI and UCB with a fixed weight
    # We weight EI slightly higher as it directly targets regret reduction,
    # but UCB helps escape local optima.
    weights_ei = 0.7
    weights_ucb = 0.3
    combined_score = weights_ei * ei_norm + weights_ucb * ucb_norm
    
    # 5. Check deadline (should never trigger, but just in case)
    if time.time() - start_time > deadline - 5:
         # Return the best result found so far (combined_score)
         # Since it's fast, we just return it.
         pass
         
    return combined_score
# EVOLVE-BLOCK-END
