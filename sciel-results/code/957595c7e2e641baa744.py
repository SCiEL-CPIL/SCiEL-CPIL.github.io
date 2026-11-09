"""Cost-aware acquisition for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: kill the program before 330s wall-clock.
_DEADLINE_SECONDS = 320.0  # safety margin
_start_time = time.time()


def _time_remaining():
    return _DEADLINE_SECONDS - (time.time() - _start_time)


def _expected_improvement(mu, sigma, f_best):
    """Compute Expected Improvement for minimization."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _ucb(mu, sigma, f_best, beta=2.0):
    """Compute Upper Confidence Bound for minimization."""
    return mu - beta * sigma


def _probability_of_improvement(mu, sigma, f_best, xi=0.01):
    """Compute Probability of Improvement for minimization."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu - xi) / sigma_safe
    return norm.cdf(z)


def _kmeans_select(candidates, scores, k=5):
    """Select k diverse candidates using a simple greedy approach based on scores."""
    if len(candidates) <= k:
        return np.arange(len(candidates))
    
    # Normalize scores to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-12:
        return np.argsort(scores)[-k:]
    
    scores_norm = (scores - s_min) / (s_max - s_min)
    
    # Greedy selection: pick highest score, then most distant from selected
    selected = [np.argmax(scores_norm)]
    selected_set = set(selected)
    
    for _ in range(k - 1):
        # For each unselected candidate, compute min distance to selected
        min_dists = np.full(len(candidates), np.inf)
        for s_idx in selected_set:
            dists = np.abs(candidates - candidates[s_idx])
            min_dists = np.minimum(min_dists, dists)
        
        # Weight by score and distance
        weights = scores_norm * (1.0 + min_dists)
        weights[list(selected_set)] = -np.inf
        
        next_idx = np.argmax(weights)
        selected.append(next_idx)
        selected_set.add(next_idx)
    
    return np.array(selected)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Check deadline
    if _time_remaining() < 5.0:
        # Emergency: return EI/cost quickly
        eps = 1e-9
        z = (f_best - mu) / (sigma + eps)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0) / (cost + eps)
    
    n = len(mu)
    
    # Handle edge cases
    if n == 0:
        return np.array([])
    
    # Ensure all inputs are finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Compute multiple acquisition functions
    ei = _expected_improvement(mu, sigma, f_best)
    ucb = _ucb(mu, sigma, f_best, beta=2.0)
    pi = _probability_of_improvement(mu, sigma, f_best, xi=0.01)
    
    # Normalize each acquisition to [0, 1] for blending
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    ei_norm = normalize(ei)
    ucb_norm = normalize(-ucb)  # Negate because lower UCB is better
    pi_norm = normalize(pi)
    
    # Blend: weight EI most heavily, then UCB, then PI
    # This balances exploitation (EI) with exploration (UCB, PI)
    blended = 0.5 * ei_norm + 0.3 * ucb_norm + 0.2 * pi_norm
    
    # Cost-aware: divide by cost
    final_score = blended / cost
    
    # Ensure finite output
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
