"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Gaussian Expected Improvement
    with an adaptive uncertainty bonus and a mild lower-bound term.
    
    This combines the exploitation strength of EI with an exploration bonus
    that is large early (when f_best is high) and small late (when f_best is low),
    plus a mild LCB component to encourage exploration in high-uncertainty regions.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    # Deadline guard: return quickly if we're close to the time limit
    # The acquisition function should be fast, but let's be safe.
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Sanitize sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    sigma_safe = np.nan_to_num(sigma_safe, nan=1e-12, posinf=1e6, neginf=1e-12)
    
    # Compute Gaussian Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Numerical stability
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Adaptive exploration bonus
    # We want more exploration when f_best is high (early) and less when low (late)
    # Use a sigmoid-like scaling: beta is large when f_best is large
    # beta = base + amplitude * (f_best / (f_best + scale))
    # When f_best = 50: beta ~ base + amplitude * 0.98
    # When f_best = 1: beta ~ base + amplitude * 0.5
    # When f_best = 0.1: beta ~ base + amplitude * 0.09
    
    # The Beale function minimum is 0, and typical values in the domain are 0-100
    # Use a scale of 5.0 so the transition happens around f_best = 5
    base_beta = 0.1
    amplitude_beta = 0.8
    scale_beta = 5.0
    
    beta = base_beta + amplitude_beta * (f_best / (f_best + scale_beta))
    
    # Exploration bonus
    exploration_bonus = beta * sigma_safe
    
    # Mild LCB component to encourage exploration in high-uncertainty regions
    # LCB = -mu + gamma * sigma
    # We use a small gamma to not dominate the EI term
    gamma = 0.3 * (f_best / (f_best + scale_beta))  # Scales with exploration need
    lcb_term = gamma * sigma_safe
    
    # Combine: EI + exploration bonus + LCB term
    # EI already has a sigma component, so we're adding on top
    scores = ei + exploration_bonus + lcb_term
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
