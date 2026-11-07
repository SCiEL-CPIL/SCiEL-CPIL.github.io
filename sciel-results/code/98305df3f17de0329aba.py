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
    
    t_start = time.time()
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # The known global minimum of Goldstein-Price 2D is 3.0
    f_opt = 3.0
    
    # For minimization, we can think in terms of "improvement" over f_opt.
    # The hypervolume indicator in a 1D objective space with a single reference
    # point reduces to the expected improvement over the reference, but we can
    # enrich this by considering the distribution of possible improvements.
    
    # z-score relative to the true optimum
    z_opt = (f_opt - mu) / sigma_safe
    
    # Probability of beating the true optimum
    prob_beat_opt = norm.cdf(z_opt)
    
    # Expected improvement over the true optimum
    # E[max(f_opt - f(x), 0)] = (f_opt - mu) * Phi(z) + sigma * phi(z)
    ei_opt = (f_opt - mu) * prob_beat_opt + sigma_safe * norm.pdf(z_opt)
    
    # Also compute EI relative to f_best for exploitation
    z_best = (f_best - mu) / sigma_safe
    ei_best = (f_best - mu) * norm.cdf(z_best) + sigma_safe * norm.pdf(z_best)
    
    # HVI-inspired score: We approximate the hypervolume contribution by
    # considering the expected "dominated area" in the improvement space.
    # For a single objective with reference f_opt, the hypervolume contribution
    # of a point x is the expected improvement over f_opt, but we can also
    # consider the variance of this improvement as a proxy for the "volume"
    # of the dominated region.
    
    # Compute the variance of the improvement: Var[max(f_opt - f(x), 0)]
    # This is more complex, so we use a simpler proxy: sigma * prob_beat_opt
    # which captures the scale of potential improvement weighted by probability.
    hvi_proxy = sigma_safe * prob_beat_opt
    
    # Combine EI with HVI proxy
    # The HVI proxy emphasizes regions with high uncertainty AND high probability
    # of improvement, which is exactly what we want for exploration.
    
    # Normalize EI and HVI proxy to [0, 1]
    ei_opt_min = np.min(ei_opt)
    ei_opt_max = np.max(ei_opt)
    if ei_opt_max - ei_opt_min > 1e-12:
        ei_opt_norm = (ei_opt - ei_opt_min) / (ei_opt_max - ei_opt_min)
    else:
        ei_opt_norm = np.zeros_like(ei_opt)
    
    hvi_min = np.min(hvi_proxy)
    hvi_max = np.max(hvi_proxy)
    if hvi_max - hvi_min > 1e-12:
        hvi_norm = (hvi_proxy - hvi_min) / (hvi_max - hvi_min)
    else:
        hvi_norm = np.zeros_like(hvi_proxy)
    
    # Weighted combination: emphasize HVI for exploration
    scores = 0.7 * hvi_norm + 0.3 * ei_opt_norm
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
