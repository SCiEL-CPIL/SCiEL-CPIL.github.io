# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Knowledge Gradient approximation with
    a UCB exploration bonus, tuned for Weierstrass 1D minimization.
    
    Uses a hybrid: 
      - Base: Expected Improvement (exploitative)
      - Bonus: scaled UCB term for exploration
      - Global view: subtract a small penalty based on how close mu is to f_best
        relative to sigma (encourages moving away from already-explored good regions
        slightly when uncertainty is high)
    """
    # Deadline guard: if we've used most of the 300s budget, fall back to pure exploitation
    elapsed = time.time() - _start_time
    if elapsed > 280.0:
        # Pure mean minimization - fastest and safest
        return -mu
    
    # Ensure numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # z-score: how many std devs below f_best is the posterior mean
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement (minimization version)
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # UCB component: -mu + beta * sigma
    # Use a moderate beta to balance exploration and exploitation
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # Hybrid: weight EI and UCB
    # When EI is large (clear improvement), lean on EI
    # When EI is small (uncertain), lean on UCB
    # Use a sigmoid-like weighting based on relative EI magnitude
    ei_scale = np.maximum(np.abs(f_best - mu), 1e-12)
    relative_ei = ei / ei_scale  # normalized EI, roughly in [0, 1]
    
    # Weight: more weight on EI when it's confidently positive
    weight_ei = 0.6 + 0.4 * relative_ei  # between 0.6 and 1.0
    weight_ucb = 1.0 - weight_ei
    
    scores = weight_ei * ei + weight_ucb * ucb
    
    # Additional exploration bonus for regions far from current best
    # This helps escape local minima in the rugged Weierstrass landscape
    # Penalize candidates whose mu is already close to f_best (already explored)
    # by reducing their score when sigma is also low
    exploration_bonus = 0.1 * sigma_safe * np.exp(-np.abs(z_clipped) * 0.5)
    
    scores += exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
