"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition with a finite Gaussian mixture model.
    
    Constructs a GMM of the posterior by discretizing the normalized uncertainty
    distribution, then scores each candidate by its expected value under a single
    random draw from this GMM. This fundamentally changes the search from a
    deterministic scoring rule to a stochastic one that naturally balances
    exploration and exploitation.
    
    Parameters
----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    t_start = time.time()
    
    n = mu.shape[0]
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Theoretical global minimum of Hartmann 6D function
    f_min = -3.32237
    regret = max(f_best - f_min, 0.0)
    
    # Check time budget early
    if time.time() - t_start > 290:
        # Fallback to simple LCB if we're running out of time
        scores = -mu + 1.5 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Construct a finite GMM of the posterior
    # Use percentiles of sigma to define component variances
    # Number of mixture components
    K = min(5, max(3, n // 500))  # 3 to 5 components
    
    # Define component means and variances based on the sigma distribution
    # We'll create components that cover the range of uncertainty
    sigma_vals = np.sort(sigma_safe)
    
    # Choose K representative sigma values (quantiles)
    quantiles = np.linspace(0.1, 0.9, K)
    component_scales = np.quantile(sigma_vals, quantiles)
    
    # Ensure scales are positive and distinct
    component_scales = np.maximum(component_scales, 1e-8)
    
    # Component weights: uniform for simplicity, could be adapted
    weights = np.ones(K) / K
    
    # Component means: centered around the mean of mu, with some spread
    # We want components to explore different regions
    mu_mean = np.mean(mu)
    mu_std = np.std(mu) + 1e-8
    
    # Create component means with some spread
    component_means = mu_mean + mu_std * np.linspace(-0.5, 0.5, K)
    
    # Single random draw from the GMM
    # Choose which component to draw from
    comp_idx = np.random.choice(K, p=weights)
    
    # Draw a scalar value from the chosen component's Gaussian
    # This represents a sample from the posterior at a "typical" point
    draw = component_means[comp_idx] + component_scales[comp_idx] * np.random.normal()
    
    # Now score each candidate based on how well it matches this draw
    # For minimization, we want candidates where mu is low and sigma is high
    # relative to the draw
    
    # Score: how likely is it that this candidate has a value lower than the draw?
    # P(f(x) < draw | mu, sigma) = Phi((draw - mu) / sigma)
    z = (draw - mu) / sigma_safe
    scores = norm.cdf(z)
    
    # Add a small bonus for high uncertainty to encourage exploration
    sigma_norm = sigma_safe / (np.max(sigma_safe) + 1e-12)
    scores += 0.1 * sigma_norm
    
    # Check time budget
    if time.time() - t_start > 295:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
