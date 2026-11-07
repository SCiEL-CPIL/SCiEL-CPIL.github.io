"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Combined Acquisition Function: Ensemble of UCB and Probability of Improvement (PI).
    
    - UCB (beta=2.0): Provides strong exploration in high-uncertainty regions.
    - PI (xi=0.5): Provides probabilistic exploitation with an optimism bias to encourage 
      exploration of regions that have a reasonable chance of improving the best value.
    
    The scores are normalized and blended to balance exploration and exploitation.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Upper Confidence Bound (UCB)
    beta = 2.0
    ucb_scores = beta * sigma_safe - mu
    
    # Normalize UCB scores to [0, 1] to make them comparable with PI
    # We use min-max normalization
    ucb_min = np.min(ucb_scores)
    ucb_max = np.max(ucb_scores)
    ucb_range = ucb_max - ucb_min
    
    if ucb_range > 1e-9:
        ucb_norm = (ucb_scores - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_scores)
    
    # 2. Probability of Improvement (PI) with optimism bias xi
    # P(f(x) < f_best - xi) is not quite right for minimization with optimism.
    # We want P(f(x) < f_best - xi) ? No, we want to encourage finding f(x) < f_best.
    # Optimism bias xi shifts the threshold: we are interested in P(f(x) < f_best - xi) 
    # if we want to be very optimistic? 
    # Actually, standard PI for minimization is P(f(x) < f_best).
    # Adding xi in the numerator: (f_best - mu + xi) / sigma 
    # This increases z, thus increasing PI. It makes the acquisition more exploratory.
    xi = 0.5
    z = (f_best - mu + xi) / sigma_safe
    
    # Clip z to prevent numerical issues in CDF
    z_clipped = np.clip(z, -10, 10)
    pi_scores = norm.cdf(z_clipped)
    
    # 3. Combine scores
    # Weighting: 50% UCB, 50% PI
    # UCB is good for finding new basins. PI is good for refining.
    # Given the small budget (25 evals), we need strong exploration.
    # Let's weight UCB slightly higher.
    w_ucb = 0.6
    w_pi = 0.4
    
    combined_scores = w_ucb * ucb_norm + w_pi * pi_scores
    
    # Ensure finite scores
    combined_scores = np.nan_to_num(combined_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return combined_scores
# EVOLVE-BLOCK-END
