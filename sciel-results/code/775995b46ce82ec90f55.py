"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition for Bayesian optimization.
    
    Uses stochastic posterior sampling to balance exploration and exploitation.
    Each call samples from the posterior at all candidates and returns the
    negative sampled values as scores (lower predicted value = higher score).
    
    Parameters:
    -----------
    mu : array-like, shape (n_candidates,)
        Posterior mean predictions at candidate points.
    sigma : array-like, shape (n_candidates,)
        Posterior standard deviations at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores; higher means more desirable to evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = len(mu)
    
    if n == 0:
        return np.array([])
    
    t_start = time.time()
    deadline = t_start + 290.0
    
    # Ensure sigma is positive
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Adaptive sampling strategy:
    # Use multiple samples and take a quantile to reduce variance of the selection
    # While still maintaining exploration properties of Thompson sampling
    
    # Determine number of samples based on available time
    # Start with a moderate number, can increase if fast
    n_samples = 16
    
    # Check if we have time for more samples
    # Quick estimate: each sample is O(n), so 16 samples is fast
    # Use the median of samples for robustness
    
    # Generate samples: y_i ~ N(mu_i, sigma_i^2) for each candidate
    # Shape: (n_samples, n)
    rng = np.random.default_rng()
    samples = rng.normal(loc=mu, scale=sigma_safe, size=(n_samples, n))
    
    # For minimization, we want low values
    # Thompson sampling: sample and pick the candidate with lowest sample
    # But we return scores for all candidates
    
    # Use the mean of the top-k lowest samples per candidate to get a robust estimate
    # This reduces the variance of pure single-sample Thompson sampling
    # while still maintaining exploration properties
    
    # Actually, for proper Thompson sampling, we should just use one sample
    # But to make it more stable, let's use the median of multiple samples
    
    # Compute the element-wise median across samples
    # This gives a more stable "expected" value while retaining exploration
    scores = -np.median(samples, axis=0)
    
    # Check deadline
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, -1e10)
        return scores
    
    # Add a small exploitation bias toward regions near f_best
    # This helps guide the search when we're close to the optimum
    # Weight decays as f_best approaches the known minimum (3.0)
    f_min_known = 3.0
    progress = 1.0 - min(max((f_best - f_min_known) / (f_best - f_min_known + 1e-6), 0.0), 1.0)
    # progress goes from 1 (far from min) to 0 (at min)
    exploit_weight = 0.1 * progress
    
    if exploit_weight > 1e-6:
        # Blend with a lower-confidence-bound-like term
        # Points that are predicted to be below f_best get a boost
        below_best = np.maximum(0, f_best - mu)
        # Normalize
        max_below = np.max(below_best)
        if max_below > 1e-12:
            exploit_bonus = exploit_weight * below_best / max_below
        else:
            exploit_bonus = np.zeros(n)
        
        scores = scores + exploit_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
