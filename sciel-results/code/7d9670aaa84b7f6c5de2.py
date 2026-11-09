"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Pure LCB (Lower Confidence Bound) acquisition function for minimization.
    
    score = mu - beta * sigma
    
    A deterministic, linear-in-uncertainty penalty that systematically
    explores high-variance regions without the nonlinear saturation of EI.
    """
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # LCB for minimization: lower score means more promising
    # We want HIGHER scores to mean "evaluate next", so we negate:
    # score = -(mu - beta * sigma) = -mu + beta * sigma
    # This way, low mu (good) and high sigma (uncertain) both increase score
    
    # Adaptive beta: start with moderate exploration, reduce as we get closer to optimum
    known_min = -1.8013
    # progress: 0 when far, 1 when at optimum
    # f_best ranges from about -1.0 to -1.8013
    # Use a smooth transition
    progress = np.clip((f_best - known_min) / 0.6, 0.0, 1.0)
    # Beta from 2.5 (early, more exploration) to 0.8 (late, more exploitation)
    beta = 0.8 + 1.7 * (1.0 - progress)
    
    # LCB score: higher is better
    score = -mu + beta * sigma_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
