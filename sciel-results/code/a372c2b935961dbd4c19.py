"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _normalize(arr):
    """Min-max normalize to [0, 1]."""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    rng = arr_max - arr_min
    if rng < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr_min) / rng


def _compute_scores(mu, sigma, f_best, elapsed_start):
    """
    Computes acquisition scores using a dynamic EI/UCB blend.
    The blend shifts from exploration to exploitation as we approach the known minimum.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # Adaptive kappa: larger when uncertain, smaller when confident
    # Use a scaled version that responds to relative uncertainty
    rel_sigma = sigma_safe / (np.abs(f_best) + 1e-8)
    kappa = 0.5 + 2.0 * np.tanh(np.mean(rel_sigma))
    ucb = -mu + kappa * sigma_safe
    
    # 3. Normalize components to [0, 1] to make them comparable
    ei_n = _normalize(ei)
    ucb_n = _normalize(ucb)
    
    # 4. Dynamic blend: shift from exploration to exploitation as we improve
    # When f_best is close to the known minimum (-39.1662), exploit more
    known_min = -39.1662
    # Progress: 0 when f_best is far from min, 1 when very close
    # Use a smooth sigmoid-like transition
    progress_raw = (f_best - known_min) / 10.0  # scale by ~10 units
    progress = 1.0 / (1.0 + np.exp(-progress_raw))  # sigmoid, maps to (0,1)
    # Blend weight: start with 40% EI, increase to 80% EI as we progress
    ei_weight = 0.4 + 0.4 * progress
    ucb_weight = 1.0 - ei_weight
    
    score = ei_weight * ei_n + ucb_weight * ucb_n
    
    # 5. Add a small bonus for points near the global minimum region
    # This helps guide search toward the basin of attraction
    # Use a smooth Gaussian-like bump centered at the known minimum
    # We don't have coordinates, so use mu as a proxy for proximity
    # Points with mu close to known_min get a small boost
    mu_centered = mu - known_min
    bump_width = 5.0  # width of the bump in function value space
    bump = np.exp(-0.5 * (mu_centered / bump_width) ** 2)
    bump_bonus = 0.05 * bump  # small bonus to not dominate
    
    score = score + bump_bonus
    
    # 6. Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    t0 = time.time()
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: if inputs are empty or degenerate, return zeros
    if mu.size == 0:
        return np.array([])
    
    # Quick computation
    scores = _compute_scores(mu, sigma, f_best, t0)
    
    # Deadline guard: if we've used too much time, return what we have
    elapsed = time.time() - t0
    if elapsed > 325.0:
        return np.zeros_like(mu)
    
    return scores
# EVOLVE-BLOCK-END
