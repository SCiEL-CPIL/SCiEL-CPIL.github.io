"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute constants for speed
_EPS = 1e-9
_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a quantile-based Probability of Improvement (POI) acquisition function
    that ranks candidates by their likelihood of beating the incumbent.
    This decouples selection pressure from the absolute scale of the objective.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points (in objective units)
    sigma : ndarray, shape (2000,)
        GP posterior std at candidate points (in objective units)
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next
    """
    # Deadline guard: return early if we're running too long
    if (time.time() - _START_TIME) > (_DEADLINE - 5.0):
        # Fallback: simple UCB with moderate exploration
        sigma_safe = np.maximum(sigma, _EPS)
        return -mu + 0.5 * sigma_safe
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure finite values and positive sigma
    sigma_safe = np.maximum(sigma, _EPS)
    
    # Compute z-score for POI: P(f < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in CDF
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # To make the scores more discriminative and add a slight exploration bias,
    # we can blend POI with a small amount of UCB-like term.
    # However, the directive asks for a structural change to POI.
    # Let's keep it pure POI but ensure it's well-scaled.
    
    # POI is in [0, 1]. Higher is better.
    # We can return POI directly, but let's add a tiny exploration bonus
    # for points with high uncertainty, just to avoid ties and encourage
    # slight exploration in early stages.
    
    # Adaptive exploration weight:
    # When f_best is large (early), we might want to explore more.
    # When f_best is small (late), we want to exploit.
    # But POI already handles this: if f_best is large, z is large for most points,
    # so POI is close to 1 for many points. If f_best is small, only points with
    # very low mu will have high POI.
    
    # Let's use a simple blend:
    # score = poi + alpha * (sigma / max_sigma)
    # where alpha is small, like 0.01, to break ties and encourage slight exploration.
    
    sigma_max = np.max(sigma_safe) + _EPS
    exploration_bonus = 0.01 * (sigma_safe / sigma_max)
    
    scores = poi + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
