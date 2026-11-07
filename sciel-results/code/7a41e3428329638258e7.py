"""Rewritten acquisition using Thompson Sampling with rank-based uncertainty bonus."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition with adaptive exploration.
    
    Strategy:
    1. Draw one posterior sample y ~ N(mu, sigma) at each candidate.
    2. Compute a rank-based uncertainty bonus that encourages exploration
       in regions with high posterior variance, but gated by the promise
       of the sampled value.
    3. Return scores where higher means "evaluate next" (i.e., we want low
       function values, so we negate the sampled value and add a bonus).
    
    The key insight: pure Thompson Sampling can be too exploratory early on
    or too exploitative later. By blending the sampled value with a rank-based
    uncertainty bonus, we get adaptive exploration that decouples from the
    absolute magnitude of GP variance.
    """
    n = len(mu)
    
    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: draw one sample from the posterior at each candidate
    # y_i ~ N(mu_i, sigma_i)
    rng = np.random.default_rng()
    y_samples = mu + sigma_safe * rng.standard_normal(n)
    
    # Compute rank of each candidate's sampled value (lower is better)
    # rank 0 = best (lowest sampled value), rank n-1 = worst
    ranks = np.argsort(np.argsort(y_samples))
    
    # Normalize ranks to [0, 1] where 0 = best, 1 = worst
    norm_ranks = ranks / (n - 1) if n > 1 else np.zeros(n)
    
    # Compute a promise score: how much better this candidate is than f_best
    # in terms of the sampled value. Higher promise = more attractive.
    # We use (f_best - y) / (sigma + eps) as a z-score of promise
    z_promise = (f_best - y_samples) / (sigma_safe + 1e-10)
    
    # The uncertainty bonus should be high when:
    # 1. The candidate has high posterior variance (sigma is large)
    # 2. The candidate is not already clearly worse than f_best (promise is not too negative)
    # 3. We haven't already explored this region too much (captured by rank)
    
    # Scale sigma to a common range for the bonus
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    if sigma_max > sigma_min:
        sigma_norm = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros(n)
    
    # The bonus: high when sigma is large and the promise is not too negative
    # We gate the bonus by the promise: if y is much worse than f_best, don't explore
    # Use a soft gate: bonus is scaled by sigmoid of promise
    gate = 1.0 / (1.0 + np.exp(-0.5 * z_promise))
    
    # Uncertainty bonus: encourages exploring high-variance regions
    # that still have some promise
    uncertainty_bonus = sigma_norm * gate
    
    # Combine: primary signal is the negative sampled value (we want low values)
    # Secondary signal is the uncertainty bonus
    # We need to normalize the sampled values to be comparable with the bonus
    
    # Normalize y_samples to [0, 1] range (invert so lower y = higher score)
    y_min = np.min(y_samples)
    y_max = np.max(y_samples)
    if y_max > y_min:
        y_norm = (y_samples - y_min) / (y_max - y_min)
    else:
        y_norm = np.zeros(n)
    
    # Invert: lower y_samples should give higher score
    y_score = 1.0 - y_norm
    
    # Blend: 80% exploitation (sampled value), 20% exploration (uncertainty bonus)
    # The rank-based approach ensures the bonus is relative, not absolute
    scores = 0.8 * y_score + 0.2 * uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
