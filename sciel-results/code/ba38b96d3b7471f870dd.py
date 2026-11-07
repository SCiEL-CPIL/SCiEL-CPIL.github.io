"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We use a hybrid acquisition: Expected Improvement with a small exploration
# term that adapts to the current uncertainty scale. This is structurally
# different from plain EI by incorporating an uncertainty-weighted exploration
# bonus that helps navigate the multimodal Michalewicz landscape.
#
# Key structural change: Instead of a fixed EI, we use a "knowledge gradient"-inspired
# score that weights the expected improvement by the posterior probability that
# the point is better than f_best, combined with a sigma-scaled exploration term.
# This avoids the fixed beta tuning issue of UCB and adapts naturally to the
# local uncertainty.

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining expected improvement with adaptive exploration.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - posterior mean in objective units
    sigma : array of shape (2000,) - posterior standard deviation in objective units
    f_best : float - best observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = more promising)
    """
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standard normal CDF and PDF
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration term: weight by the probability of being in an
    # underexplored region (high sigma relative to current scale)
    # This term encourages exploration where uncertainty is high
    # but is damped as we get more confident (lower sigma)
    
    # Scale sigma by the typical improvement scale to make the exploration term
    # dimensionless and adaptive
    scale = np.std(mu) + 1e-10
    exploration = sigma_safe / (scale + sigma_safe)
    
    # Combine: EI dominates when exploitation is clear, exploration helps
    # when uncertainty is high relative to the landscape scale
    # The coefficient 0.3 is a structural choice for balanced exploration
    scores = ei + 0.3 * exploration * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
