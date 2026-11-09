# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement (EI) for minimization.
    mu, sigma : (2000,) arrays, sigma is the GP posterior std (objective units)
    f_best    : lowest observed objective value
    Returns 2000 finite scores; higher => evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-9)

    # EI for minimization: maximize f_best - f(x)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10.0, 10.0)
    
    scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Ensure finiteness and a finite spread (avoid all-equal scores).
    scores = np.where(np.isfinite(scores), scores, 0.0)
    if np.allclose(scores, scores[0]):
        # Fallback: rank by sigma (explore) with tiny noise-free tiebreak
        scores = sigma + 1e-12 * np.arange(len(sigma))

    return scores
# EVOLVE-BLOCK-END
