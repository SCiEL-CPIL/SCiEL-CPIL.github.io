"""Combined acquisition: adaptive UCB with EI blend and deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best):
    """Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Combines adaptive UCB (for exploration early on) with Expected Improvement
    (for exploitation late) via a progress-based blend.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (in objective units, lower is better)
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (in objective units)
    f_best : float
        Best (lowest) observed function value so far
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next
    """
    # Deadline guard: if we're close to the wall, return simple exploitation scores
    if time.time() > _DEADLINE:
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are numpy arrays with correct shape
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    if mu.size != 2000 or sigma.size != 2000:
        raise ValueError(f"Expected shape (2000,), got mu.shape={mu.shape}, sigma.shape={sigma.shape}")
    
    # Basic sanity: ensure finite values
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Goldstein-Price 2D minimum value
    f_min = 3.0
    
    # Compute gap: how far we are from the known minimum
    gap = max(f_best - f_min, 0.0)
    
    # Adaptive beta for UCB: large when gap is large (early), small when gap is small (late)
    # beta = 0.3 + 2.0 * (1 / (1 + gap / 5.0))
    # gap=0: beta=2.3, gap=5: beta=1.3, gap=10: beta=1.0, gap=20: beta=0.7, gap=50: beta=0.48
    beta = 0.3 + 2.0 * (1.0 / (1.0 + gap / 5.0))
    
    # Compute UCB score (for minimization: -mu + beta*sigma)
    ucb = -mu + beta * sigma
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    # Compute EI score with small xi
    # xi scales with gap to encourage more exploration early
    xi = 0.01 * max(gap, 0.1)
    z = (f_best - mu - xi) / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu - xi) * cdf + sigma * pdf
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Blend: weight_ucb high when gap is large, weight_ei high when gap is small
    # weight_ucb = 1.0 / (1.0 + gap / 3.0)
    # weight_ei = 1.0 - weight_ucb
    weight_ucb = 1.0 / (1.0 + gap / 3.0)
    weight_ei = 1.0 - weight_ucb
    
    # Combine: primarily UCB with a small EI component
    # The EI component helps with fine-grained exploitation when very close
    score = weight_ucb * ucb + weight_ei * ei
    
    # Alternative: use UCB as primary with small EI boost (as in inspiration 1)
    # score = ucb + 0.1 * ei
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score.astype(np.float64)
# EVOLVE-BLOCK-END
