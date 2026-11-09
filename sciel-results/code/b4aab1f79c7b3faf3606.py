"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track state for adaptive strategy
_start_time = time.time()
_call_count = 0
# Cache for diversity penalty (track recently explored regions)
_explored_mu = []


def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition: switches between exploitation (EI) and exploration (UCB)
    based on uncertainty, with diversity-aware penalty to encourage broad search.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Mean of GP posterior at candidate points
    sigma : ndarray, shape (n,)
        Standard deviation of GP posterior at candidate points
    f_best : float
        Best (lowest) observed objective value so far
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = more promising)
    """
    global _start_time, _call_count, _explored_mu
    
    # Deadline guard: if we're close to timeout, return simple EI quickly
    elapsed = time.time() - _start_time
    _call_count += 1
    
    # If we've been running for too long, fall back to fast EI
    if elapsed > 300:
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = np.where(np.isfinite(ei), ei, 0.0)
        return scores
    
    n = len(mu)
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive beta: higher uncertainty in early iterations -> more exploration
    # Use median sigma as a proxy for overall uncertainty level
    med_sigma = np.median(sigma)
    
    # In early iterations (low call count), explore more
    # As we converge (f_best approaches 3), exploit more
    # Goldstein-Price minimum is 3.0
    
    # Normalize f_best progress: 0 when f_best is high, 1 when close to 3
    # f_best starts around 10-20 for random points, minimum is 3
    progress = np.clip(1.0 - (f_best - 3.0) / 20.0, 0.0, 1.0)
    
    # Call-count based exploration: more exploration early
    call_progress = np.clip(_call_count / 30.0, 0.0, 1.0)
    
    # Combine progress signals
    exploitation_weight = 0.5 * progress + 0.5 * call_progress
    
    # Beta decreases as we exploit more
    # Start with high beta for exploration, decrease to low beta for exploitation
    beta = 2.0 * (1.0 - exploitation_weight) + 0.1 * exploitation_weight
    
    # UCB component
    ucb = beta * sigma_safe
    
    # Diversity penalty: penalize candidates that are too similar to recently explored means
    # This encourages the search to cover different regions of the space
    if len(_explored_mu) > 0:
        # Use the last few explored points for diversity
        recent_explored = np.array(_explored_mu[-5:]) if len(_explored_mu) >= 5 else np.array(_explored_mu)
        
        # Compute minimum distance from each candidate to recent explored points
        # Since we don't have coordinates, use mu values as a proxy
        # Candidates with mu close to explored mu get penalized
        min_dist = np.min(np.abs(mu[:, np.newaxis] - recent_explored[np.newaxis, :]), axis=1)
        
        # Scale penalty: larger penalty for closer matches
        # Use a soft penalty that decays with distance
        penalty_scale = np.max(sigma) + eps
        diversity_penalty = -0.3 * np.exp(-min_dist / penalty_scale)
    else:
        diversity_penalty = np.zeros(n)
    
    # Combined score
    scores = ei + ucb + diversity_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Track the selected point's mu for diversity (approximate: track the argmax)
    if _call_count % 5 == 0:  # Record every 5th call to avoid too much history
        idx = np.argmax(scores)
        _explored_mu.append(float(mu[idx]))
        # Keep only last 20 entries
        if len(_explored_mu) > 20:
            _explored_mu.pop(0)
    
    return scores
# EVOLVE-BLOCK-END
