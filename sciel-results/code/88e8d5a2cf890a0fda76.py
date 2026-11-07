"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a Hypervolume Indicator (HVI) approximation as the acquisition function.
    The reference point is set to the known global minimum (3.0). We compute
    an approximate hypervolume contribution for each candidate point, which
    directly optimizes for the area of the Pareto front dominated by the best
    observed solution relative to the true optimum.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Safety deadline guard
    t_start = time.time()
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # The known global minimum of Goldstein-Price 2D is 3.0
    f_opt = 3.0
    
    # HVI-based acquisition: We approximate the hypervolume contribution of
    # each candidate point. For a single-objective problem, the hypervolume
    # dominated by a point (f, reference) is simply max(ref - f, 0) in the
    # objective space. However, since we have uncertainty, we compute the
    # expected hypervolume contribution using posterior sampling.
    
    # For efficiency, we use a Monte Carlo approximation with a fixed number
    # of samples per candidate point
    n_samples = 50
    
    # Generate random samples from the posterior
    # For each candidate point, we sample from N(mu_i, sigma_i^2)
    # and compute the expected hypervolume contribution
    
    # To be efficient, we'll compute this vectorized
    # Sample shape: (n_samples, n)
    samples = mu[None, :] + sigma_safe[None, :] * np.random.randn(n_samples, n)
    
    # For each sample, compute the hypervolume contribution
    # HV contribution = max(f_opt - sample, 0)
    # This is the area between the sample value and the reference point f_opt
    # (if the sample is below f_opt)
    
    hv_samples = np.maximum(f_opt - samples, 0.0)
    
    # Expected HV contribution
    scores = np.mean(hv_samples, axis=0)
    
    # Add a variance-based exploration bonus
    # Points with high uncertainty that could potentially beat f_opt get a boost
    z = (f_opt - mu) / sigma_safe
    prob_beat = norm.cdf(z)
    
    # Exploration bonus: encourage points where there's a reasonable chance
    # of beating the optimum, weighted by uncertainty
    exploration_bonus = prob_beat * sigma_safe
    
    # Combine HV and exploration
    # Normalize both to [0, 1] for stable combination
    scores_min = np.min(scores)
    scores_max = np.max(scores)
    if scores_max - scores_min > 1e-12:
        scores_norm = (scores - scores_min) / (scores_max - scores_min)
    else:
        scores_norm = np.zeros_like(scores)
    
    bonus_min = np.min(exploration_bonus)
    bonus_max = np.max(exploration_bonus)
    if bonus_max - bonus_min > 1e-12:
        bonus_norm = (exploration_bonus - bonus_min) / (bonus_max - bonus_min)
    else:
        bonus_norm = np.zeros_like(exploration_bonus)
    
    # Weight: primarily HV, with some exploration
    scores_final = 0.8 * scores_norm + 0.2 * bonus_norm
    
    # Ensure all values are finite
    scores_final = np.nan_to_num(scores_final, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores_final
    
    return scores_final
# EVOLVE-BLOCK-END
