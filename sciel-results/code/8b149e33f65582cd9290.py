"""Rewritten acquisition for Bayesian optimization of Hartmann-3D using Knowledge Gradient (KG)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE_SECONDS = 330.0
_SAFETY_MARGIN = 10.0

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using the Knowledge Gradient (KG) criterion.
    For a single candidate, the one-step KG is equivalent to Expected Improvement (EI).
    We use the closed-form analytical solution for speed and precision.
    
    KG(x) = E[min(f_best, f(x))] - f_best
    Since we minimize, we maximize the reduction: f_best - E[min(f_best, f(x))].
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    _ensure_start_time()
    
    # Check deadline guard
    if _elapsed() > _DEADLINE_SECONDS - _SAFETY_MARGIN:
        # Fallback to simple EI-based score quickly
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical overflow in exp() inside pdf
    z_clipped = np.clip(z, -10, 10)
    
    # Closed-form Knowledge Gradient (equivalent to Expected Improvement for minimization)
    # KG = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: (f_best - mu) is negative if mu < f_best (improvement expected).
    # Wait, if mu < f_best, z > 0.
    # Let's check signs.
    # If mu < f_best, we expect improvement.
    # z = (f_best - mu) / sigma > 0.
    # Phi(z) > 0.5.
    # (f_best - mu) is positive.
    # So (f_best - mu) * Phi(z) is positive.
    # sigma * phi(z) is positive.
    # So KG is positive.
    
    kg_scores = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure all scores are finite
    kg_scores = np.where(np.isfinite(kg_scores), kg_scores, 0.0)
    
    # Add a tiny random tie-breaker to avoid deterministic behavior in case of ties
    # This is structurally part of the "exploration" aspect of stochastic acquisition, 
    # though pure KG is deterministic. A tiny jitter helps the BO loop navigate plateaus.
    if len(kg_scores) > 0:
        # Use a fixed seed for reproducibility in the tie-breaker if needed, 
        # or just default RNG. The task allows randomness.
        rng = np.random.default_rng()
        kg_scores += rng.uniform(0, 1e-8, size=len(kg_scores))
        
    return kg_scores
# EVOLVE-BLOCK-END
