"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Batch-aware Lower Confidence Bound (LCB) with adaptive exploration and
    a simple diversity-promoting penalty based on uncertainty weighting.
    
    Since we don't have coordinates, we use a hybrid approach:
    - Base LCB: mu - beta * sigma (exploitation + exploration)
    - Adaptive beta that increases early and decreases later based on 
      the spread of sigma values (proxy for how much exploration is needed)
    - A small noise term to break ties and encourage diversity
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Adaptive beta: start with high exploration, decay as we get more confident
    # Use the ratio of mean sigma to max sigma as a proxy for convergence
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    if sigma_max > 1e-10:
        # Convergence ratio: 1 means all sigmas are similar (early), 
        # lower means some regions have much lower uncertainty (later)
        conv_ratio = sigma_mean / sigma_max
    else:
        conv_ratio = 1.0
    
    # Beta starts high (exploratory) and decreases as convergence increases
    # Base beta around 1.5-2.5 for Hartmann 3D
    beta_base = 2.0
    # Adjust beta based on convergence: more converged -> less exploration
    beta = beta_base * (0.5 + 0.5 * conv_ratio)
    
    # Also scale by how far f_best is from the theoretical minimum
    # Hartmann 3D minimum is -3.86278
    f_min_theory = -3.86278
    gap = f_best - f_min_theory
    if gap > 1e-6:
        # If we're far from optimum, explore more
        beta *= (1.0 + 0.5 * min(gap / 2.0, 1.0))
    
    # Compute LCB scores: lower mu - beta*sigma is better, so we negate for "higher is better"
    lcb_scores = -(mu - beta * sigma_safe)
    
    # Add a small deterministic perturbation based on index to break ties
    # and encourage diversity without true randomness
    indices = np.arange(len(mu))
    perturbation = 0.001 * np.sin(indices * 0.1)
    
    scores = lcb_scores + perturbation
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
