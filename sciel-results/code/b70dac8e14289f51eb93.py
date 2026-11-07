"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Pure Lower Confidence Bound (LCB) acquisition function for minimization.
    
    Strategy:
    - Use a fixed, aggressive exploration constant k=3.0.
    - Score = -mu + k * sigma.
    - This aggressively targets regions with high posterior variance (high sigma)
      while still favoring regions with lower mean predictions (low mu).
    - The Michalewicz function has an isolated global minimum surrounded by
      high-gradient ridges; uncertainty-driven exploration is key to finding it.
    - This is a deterministic, non-probabilistic approach that avoids the
      computational overhead and potential numerical issues of EI.
    """
    # Track elapsed time for safety
    t_start = time.time()
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB for minimization:
    # We want to maximize the score.
    # A high score should correspond to a point that is likely to have a low function value.
    # Lower bound of prediction: mu - k * sigma.
    # We want the "best" (lowest) possible value, i.e., maximize optimism.
    # Optimism = -(mu - k * sigma) = -mu + k * sigma.
    
    k = 3.0
    score = -mu + k * sigma_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
