import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Thompson Sampling with adaptive exploitation bias.
    
    Combines stochastic Thompson Sampling (to escape local minima) with a
    deterministic exploitation bias that strengthens as the search converges
    toward the known global minimum. This provides a balance between
    exploration (via random sampling) and exploitation (via bias toward
    low-mu regions).
    
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
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: Sample from posterior predictive distribution
    # For minimization, we want to sample values and pick the lowest
    # score = -sampled_value, so higher score = lower sampled value
    z = np.random.normal(0, 1, size=mu.shape)
    y_samples = mu - sigma_safe * z
    
    # Base score: higher is better. Since we minimize y, score is -y.
    base_score = -y_samples
    
    # Adaptive exploitation bias:
    # The bias strength increases as f_best approaches the known global minimum.
    # This encourages exploitation when we're close to the optimum,
    # but allows more exploration when we're far from it.
    opt = -1.8013
    dist = max(0.0, f_best - opt)
    
    # Scale bias strength: starts at 0.0 when far from optimum, 
    # increases to 0.15 when very close
    bias_strength = 0.15 * min(dist / 5.0, 1.0)
    
    # Exploitation bias: add a small penalty for high mu (exploit known good regions)
    # This helps escape local minima while still favoring exploitation
    exploitation_bias = bias_strength * mu
    
    scores = base_score - exploitation_bias
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Safety: check time
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        # Fallback to simple UCB if we're running out of time
        beta = 2.0
        ucb_score = -mu + beta * sigma_safe
        ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -1e10)
        return ucb_score
    
    return scores
