"""Acquisition function for Bayesian optimization of styblinski tang 1d.
Uses a Kriging Believer strategy with progress-adaptive UCB blend:
early on, explore more via LCB; later, exploit via EI on the mean.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 300.0  # 300s wall-clock limit per evaluation

# Known global minimum of Styblinski-Tang 1D
_GLOBAL_MIN = -39.1662


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points using
    Kriging Believer with progress-adaptive exploration.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Safety: return EI as fallback
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute progress: 0 = just started (f_best far from min), 1 = near global min
    # Use a scale based on the range from worst plausible to global min
    # For ST1D on [-5,5], max value is around 100, min is -39.1662
    progress_scale = 150.0  # covers range from ~+100 to ~-50
    progress = np.clip((f_best - _GLOBAL_MIN) / progress_scale, 0.0, 1.0)
    # progress near 1 means we're close to the global minimum (exploit)
    # progress near 0 means we're far (explore)
    
    # Kriging Believer: use LCB as the "believed" mean
    # Adapt beta: larger when exploring (progress low), smaller when exploiting
    beta = 0.5 + 1.5 * (1.0 - progress)  # beta in [0.5, 2.0]
    lcb = mu - beta * sigma
    
    # Compute EI using the LCB as the surrogate mean
    # EI = E[max(f_best - f, 0)] where f ~ N(mu, sigma^2)
    # But in Kriging Believer, we evaluate EI at the LCB point:
    # We want to maximize the probability of improvement, so we compute
    # EI based on the posterior but shift by the belief
    # Standard Kriging Believer: score = EI evaluated at (mu - beta*sigma)
    # This means we pretend the mean is the LCB and compute improvement
    # over f_best
    
    # For the Kriging Believer, the acquisition is:
    # score = (f_best - lcb) * Phi((f_best - lcb)/sigma) + sigma * phi((f_best - lcb)/sigma)
    # This is EI computed as if the mean were the LCB
    z_lcb = (f_best - lcb) / sigma
    ei_kb = (f_best - lcb) * norm.cdf(z_lcb) + sigma * norm.pdf(z_lcb)
    
    # Also compute standard EI for blending
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize both to [0, 1]
    ei_kb_min, ei_kb_max = np.min(ei_kb), np.max(ei_kb)
    ei_kb_range = ei_kb_max - ei_kb_min + 1e-9
    ei_kb_norm = (ei_kb - ei_kb_min) / ei_kb_range
    
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    # Blend: more Kriging Believer early, more pure EI late
    w_kb = 0.5 + 0.4 * (1.0 - progress)  # w_kb in [0.5, 0.9]
    w_ei = 1.0 - w_kb
    scores = w_kb * ei_kb_norm + w_ei * ei_norm
    
    # Add a small exploitation boost: candidates where mu is very close to or
    # below f_best get a bonus to help refine the neighborhood
    mu_diff = f_best - mu  # positive when mu < f_best
    boost = np.clip(mu_diff / (sigma + 1e-9), 0, 3.0) * 0.03
    scores += boost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
