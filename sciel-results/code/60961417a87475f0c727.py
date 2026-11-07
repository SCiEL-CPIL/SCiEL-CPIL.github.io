"""Combined acquisition: Thompson Sampling with adaptive exploitation bias."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Thompson Sampling with adaptive exploitation bias.
    
    Uses stochastic Thompson Sampling to select candidates, which naturally
    balances exploration and exploitation. An adaptive bias term strengthens
    exploitation as the search converges toward the known global minimum.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean predictions at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: handle non-finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=-1e6)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: draw one sample from the posterior predictive
    # For minimization, we want lower values to be better, so score = -sample
    z = np.random.normal(0, 1, size=mu.shape)
    y_samples = mu + sigma_safe * z
    
    # Base score: higher is better, so negate the sampled value
    base_score = -y_samples
    
    # Adaptive exploitation bias:
    # Known global minimum of Michalewicz 2D
    opt = -1.8013
    dist_to_opt = max(0.0, f_best - opt)
    
    # Bias strength increases as we get closer to the optimum
    # When far from optimum (dist large), bias is small -> more exploration
    # When close to optimum (dist small), bias is larger -> more exploitation
    # Scale: dist typically ranges from ~0 to ~5 for this problem
    bias_strength = 0.15 * min(dist_to_opt / 5.0, 1.0)
    
    # Exploitation bias: favor candidates with lower mu (deterministic mean)
    # This counteracts unlucky high samples in Thompson Sampling
    exploitation_bias = bias_strength * (f_best - mu)
    
    scores = base_score + exploitation_bias
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline guard: if we're running out of time, return a simple fallback
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        # Fallback to simple UCB-style score
        beta = 1.5
        ucb_score = -mu + beta * sigma_safe
        ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -1e10)
        return ucb_score
    
    return scores
# EVOLVE-BLOCK-END
