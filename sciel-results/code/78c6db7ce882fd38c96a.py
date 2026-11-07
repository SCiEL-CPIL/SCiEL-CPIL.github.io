"""Model-and-optimize acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE_SECONDS = 300.0
_SAFETY_MARGIN = 20.0

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _ei_scores(mu, sigma, f_best):
    """Standard Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(np.isfinite(ei), ei, 0.0)

def acquisition(mu, sigma, f_best):
    """
    Deterministic GP (DGP) acquisition: compute the minimum predictive mean
    over all candidates as a surrogate lower bound, then score each candidate
    by its expected improvement over that bound. This adapts exploration
    intensity to how confident the GP is about the global minimum.
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    _ensure_start_time()
    
    # Deadline guard: fall back to simple EI
    if _elapsed() > _DEADLINE_SECONDS - _SAFETY_MARGIN:
        return _ei_scores(mu, sigma, f_best)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # DGP: deterministic lower bound = min predictive mean across candidates
    dgp_bound = np.min(mu)
    
    # Expected improvement of each candidate over the DGP bound
    z = (dgp_bound - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei_dgp = (dgp_bound - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Combine: primary signal is EI over DGP bound,
    # plus a small exploitation term toward f_best to avoid pure exploitation
    # of already-explored regions
    ei_fb = _ei_scores(mu, sigma, f_best)
    
    # Blend: 70% DGP-EI (explore where GP thinks minimum could be),
    # 30% standard EI (exploit toward current best)
    scores = 0.7 * ei_dgp + 0.3 * ei_fb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
