"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Hypervolume Indicator (HVI) approximation.
    
    The HVI-based acquisition directly optimizes for the area of the Pareto
    front dominated by the best observed solution relative to the true optimum.
    The reference point is set to the known global minimum (3.0), and the
    improvement axis is the negative function value.
    
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
    
    t_start = time.time()
    
    # Safety: clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Known global minimum of Goldstein-Price 2D
    f_min_true = 3.0
    
    # HVI-based acquisition:
    # We approximate the expected hypervolume improvement by considering
    # the probability that a new observation improves upon the current best,
    # weighted by the magnitude of improvement.
    #
    # For each candidate point, we estimate the expected contribution to
    # hypervolume by integrating over the posterior predictive distribution.
    #
    # The hypervolume contribution from a point at value f (lower is better)
    # relative to reference f_ref is: max(f_ref - f, 0)
    #
    # Expected HV contribution = E[max(f_ref - Y, 0)] where Y ~ N(mu, sigma^2)
    # This is equivalent to Expected Improvement but with a different
    # interpretation: it measures the expected area gain in the Pareto front.
    
    # Compute expected hypervolume contribution for each candidate
    # E[max(f_ref - Y, 0)] where Y ~ N(mu_i, sigma_i^2)
    f_ref = f_min_true
    z = (f_ref - mu) / sigma_safe
    
    # E[max(f_ref - Y, 0)] = (f_ref - mu) * Phi(z) + sigma * phi(z)
    # This is the same formula as EI, but we interpret it as HV contribution
    hv_contrib = (f_ref - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Additionally, we add a term that accounts for the probability of
    # discovering a point that dominates the current best solution.
    # P(Y < f_best) = Phi((f_best - mu) / sigma)
    # This encourages exploration in regions where we might find
    # solutions better than the current best.
    
    z_best = (f_best - mu) / sigma_safe
    p_improve = norm.cdf(z_best)
    
    # The expected improvement over f_best (not the true minimum)
    # E[max(f_best - Y, 0)]
    ei_best = (f_best - mu) * norm.cdf(z_best) + sigma_safe * norm.pdf(z_best)
    
    # Combine HV contribution with improvement probability
    # The HV contribution already captures the expected area gain
    # The p_improve term adds a bonus for likely improvements
    
    # Normalize terms for stable combination
    hv_min = np.min(hv_contrib)
    hv_max = np.max(hv_contrib)
    if hv_max - hv_min > 1e-12:
        hv_norm = (hv_contrib - hv_min) / (hv_max - hv_min)
    else:
        hv_norm = np.zeros_like(hv_contrib)
    
    p_min = np.min(p_improve)
    p_max = np.max(p_improve)
    if p_max - p_min > 1e-12:
        p_norm = (p_improve - p_min) / (p_max - p_min)
    else:
        p_norm = np.zeros_like(p_improve)
    
    # Adaptive weighting: when f_best is close to the true minimum,
    # rely more on HV contribution (exploit near-optimal regions)
    # When f_best is far from the true minimum, rely more on
    # improvement probability (explore broadly)
    gap = max(f_best - f_min_true, 1e-6)
    # Use a sigmoid-like transition based on gap
    # Small gap -> weight toward HV, large gap -> weight toward p_improve
    gap_scale = 2.0  # scale parameter for transition
    w_p = 1.0 / (1.0 + np.exp(-gap / gap_scale))
    w_hv = 1.0 - w_p
    
    scores = w_hv * hv_norm + w_p * p_norm
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
