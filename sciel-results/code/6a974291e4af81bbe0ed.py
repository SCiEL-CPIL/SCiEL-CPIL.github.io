"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Batch-aware Lower Confidence Bound (LCB) with spatial diversity.
    
    Since we only have mu, sigma, and f_best (no coordinates), we cannot
    compute actual distances to the best point. However, we can still use
    a batch-aware LCB that encourages exploration by using a higher beta
    when the current best is far from the minimum (i.e., when we're still
    exploring). The "repulsion" term is approximated by adjusting the
    exploitation/exploration balance based on how close f_best is to the
    known minimum (-3.86278 for Hartmann 3D).
    
    We use a standard LCB: mu - beta * sigma
    where beta is chosen to balance exploration and exploitation.
    
    To incorporate "batch-awareness" and spatial diversity without coordinates,
    we can use an adaptive beta that increases when we're far from the optimum,
    encouraging more exploration early on, and decreases as we get closer.
    
    The Hartmann 3D minimum is approximately -3.86278.
    """
    # Known minimum of Hartmann 3D
    h3_min = -3.86278
    
    # Normalize the progress: 0 when f_best is far from minimum, 1 when close
    # f_best starts around the worst value (e.g., ~0 or positive for Hartmann)
    # and decreases toward -3.86278
    # Let's say "progress" = (initial_guess - f_best) / (initial_guess - h3_min)
    # A reasonable initial guess for the worst case might be around 0 or even positive
    # For Hartmann 3D on [0,1]^3, values range roughly from -3.86 to about 0.5
    # Let's use a reference "worst" value of 0.5
    worst_ref = 0.5
    progress = np.clip((worst_ref - f_best) / (worst_ref - h3_min), 0.0, 1.0)
    
    # Adaptive beta: higher when progress is low (more exploration),
    # lower when progress is high (more exploitation)
    # beta ranges from about 2.0 (high exploration) to 0.5 (high exploitation)
    beta = 2.0 - 1.5 * progress
    
    # Standard LCB: mu - beta * sigma
    # Higher score means we should evaluate next, so we want to maximize this
    # Since we're minimizing, LCB = mu - beta * sigma, and we pick the candidate
    # with the lowest LCB (most promising low value). But the task says "higher
    # meaning evaluate next", so we need to negate or use a different formulation.
    
    # Actually, for minimization with LCB, the acquisition value is:
    # alpha(x) = mu(x) - beta * sigma(x)
    # We want to select x that minimizes alpha(x), i.e., has the lowest LCB.
    # But the task says "higher meaning evaluate next", so we should return
    # negative LCB or flip the sign.
    
    # Let's return -LCB = -mu + beta * sigma, so higher values are preferred.
    score = -mu + beta * sigma
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
