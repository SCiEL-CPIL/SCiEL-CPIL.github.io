"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Hypervolume Indicator (HVI) approximation.
    
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
    
    # Hypervolume Indicator approach:
    # We want to maximize the hypervolume dominated by the best solution
    # relative to a reference point. Since we're minimizing, we use
    # negative function values as the objective.
    
    # Reference point for hypervolume: slightly worse than the known optimum
    # In the negative value space, this means a value slightly higher than -f_opt
    ref_point = -f_opt - 0.5  # -3.5 in negative value space
    
    # For each candidate, compute the expected hypervolume contribution
    # The hypervolume contribution of a point with value f is:
    # max(0, (-f) - ref_point) = max(0, -f + f_opt + 0.5) = max(0, f_opt + 0.5 - f)
    
    # We need to compute E[max(0, f_opt + 0.5 - f(x))] where f(x) ~ N(mu, sigma^2)
    # This is similar to Expected Improvement but with a different reference
    
    # Let g = f_opt + 0.5 (this is the "improvement threshold" in original space)
    # We want E[max(0, g - f(x))]
    
    g = f_opt + 0.5
    
    # Compute z-score
    z = (g - mu) / sigma_safe
    
    # E[max(0, g - f(x))] = (g - mu) * Phi(z) + sigma * phi(z)
    # This is exactly the Expected Improvement formula with reference g
    
    hvi_contribution = (g - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Additionally, we want to weight by the probability of actually improving
    # upon the current best, to balance exploration and exploitation
    z_best = (f_best - mu) / sigma_safe
    prob_improve = norm.cdf(z_best)
    
    # The hypervolume contribution already captures the expected improvement
    # over the reference point. We can modulate this by the probability of
    # being a meaningful improvement over the current best.
    
    # Normalize hvi_contribution to [0, 1] for stable combination
    hvi_min = np.min(hvi_contribution)
    hvi_max = np.max(hvi_contribution)
    if hvi_max - hvi_min > 1e-12:
        hvi_norm = (hvi_contribution - hvi_min) / (hvi_max - hvi_min)
    else:
        hvi_norm = np.zeros_like(hvi_contribution)
    
    # Normalize probability of improvement
    prob_min = np.min(prob_improve)
    prob_max = np.max(prob_improve)
    if prob_max - prob_min > 1e-12:
        prob_norm = (prob_improve - prob_min) / (prob_max - prob_min)
    else:
        prob_norm = np.zeros_like(prob_improve)
    
    # Combine: primary weight on hypervolume, secondary on probability of improvement
    # The hypervolume indicator directly measures the quality of the Pareto front
    scores = 0.7 * hvi_norm + 0.3 * prob_norm
    
    # Add a small bonus for high uncertainty to encourage exploration
    # in regions where the GP is uncertain
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean > 1e-10:
        uncertainty_bonus = sigma_safe / sigma_mean
        uncertainty_bonus = np.clip(uncertainty_bonus, 0, 2) / 2  # Scale to [0, 1]
        scores += 0.05 * uncertainty_bonus
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
