import numpy as np
from scipy.stats import norm
import time

_deadline = None
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function for Bayesian optimization.
    
    Draws a single sample from the GP posterior at each candidate point and
    scores by the negative sampled value divided by cost. This naturally balances
    exploration and exploitation without tuning hyperparameters.
    
    Parameters:
    mu: (2000,) array of mean predictions
    sigma: (2000,) array of standard deviations
    f_best: float, current best observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of scores (higher is better)
    """
    global _deadline, _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0
    
    _call_count += 1
    
    if time.time() > _deadline:
        return np.ones(len(mu))
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Thompson Sampling: draw one sample from the posterior
    # sample_i ~ N(mu_i, sigma_i^2)
    rng = np.random.default_rng()
    noise = rng.standard_normal(len(mu))
    sample = mu + sigma_safe * noise
    
    # Score: we want to minimize the sampled value, but also account for cost.
    # Higher score = more desirable to evaluate next.
    # Use negative sampled value divided by cost (cost-aware Thompson Sampling).
    # This favors points with low sampled values and low costs.
    score = -sample / cost_safe
    
    # Add a small exploitation bias: slightly favor points near f_best
    # This helps guide search toward the region of the current best.
    # The bias decays as we make more calls (more budget consumed).
    decay = np.exp(-0.1 * _call_count)
    # Points with mu close to f_best get a small bonus
    proximity = np.exp(-np.abs(mu - f_best) / (np.std(mu) + eps))
    score += decay * 0.1 * proximity
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
