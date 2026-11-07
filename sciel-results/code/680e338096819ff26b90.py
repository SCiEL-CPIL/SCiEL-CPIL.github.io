# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D function.
    Uses Expected Improvement (EI) with an adaptive exploration bonus.
    The exploration weight adapts based on how close we are to the optimum.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Adaptive exploration: scale sigma by a factor that depends on how close
    # we are to the optimum. When f_best is large, we explore more aggressively.
    # When f_best is small (close to 0), we rely more on exploitation.
    # Beale function has minimum 0 at (3, 0), and values can be quite large.
    # Use a smooth transition: exploration_weight starts high and decays.
    
    # Normalize f_best to get a sense of progress
    # f_best starts around 100-200 for random points on Beale, drops to 0 at optimum
    # Use a simple adaptive weight
    f_best_abs = abs(f_best)
    
    # When f_best is large, we want more exploration; when small, less
    # Use an inverse relationship with saturation
    exploration_weight = 1.0 / (1.0 + 0.1 * f_best_abs)
    
    # The exploration term: we want to favor points with high sigma,
    # but also penalize points where mu is already very high (obviously bad)
    # Use sigma * weight, but only for points where mu is not too far above f_best
    # This avoids exploring regions that are clearly terrible
    
    # Create a mask: only apply exploration bonus where mu is reasonably close
    # to f_best (within a few standard deviations)
    reasonable = mu < f_best + 2.0 * sigma_safe
    
    # Exploration bonus: scaled sigma, applied only to reasonable points
    exploration_bonus = np.where(reasonable, exploration_weight * sigma_safe, 0.0)
    
    # Total score: Higher is better
    score = ei + exploration_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
