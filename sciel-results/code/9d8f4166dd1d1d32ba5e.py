import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition with minimal exploitation bias.
    
    Primary mechanism: sample from the posterior predictive distribution
    f(x) ~ N(mu(x), sigma(x)^2) and select the point with the lowest sampled
    value. This stochastic approach naturally balances exploration and
    exploitation, enabling jumps between multimodal basins in Michalewicz 2D.
    
    A small deterministic bias toward f_best provides gentle convergence
    pressure without dominating the stochastic selection.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (N,)
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (N,)
    f_best (float): The best (lowest) objective value observed so far.
    
    Returns:
    np.ndarray: Scores for each candidate, shape (N,). Higher scores are preferred.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Thompson Sampling: sample from posterior predictive
    # f(x) ~ N(mu(x), sigma(x)^2)
    # For minimization, we want low f values, so score = -f_sample
    z = np.random.standard_normal(mu.shape)
    f_sample = mu + sigma * z
    ts_score = -f_sample

    # Minimal exploitation bias: a tiny pull toward points where the posterior
    # mean is already close to (or below) f_best. This is NOT a full EI term
    # (which would dominate and kill the stochasticity), just a gentle nudge.
    # Scale: the bias is at most ~1% of the typical score range.
    sigma_safe = np.maximum(sigma, 1e-10)
    # Normalized proximity to f_best: how much better is mu than f_best?
    # Positive when mu < f_best (promising region)
    proximity = np.maximum(f_best - mu, 0.0)
    # Scale by typical sigma to make it dimensionless and bounded
    sigma_scale = np.median(sigma_safe)
    if sigma_scale > 0:
        bias = 0.01 * proximity / sigma_scale
    else:
        bias = np.zeros_like(mu)

    score = ts_score + bias

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    return score
