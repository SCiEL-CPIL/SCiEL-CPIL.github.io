"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for time management
_start_time = None


def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time


def _time_remaining():
    """Return remaining time in seconds, with safety margin."""
    elapsed = time.time() - _get_start_time()
    # Safety margin: stop at 320s (10s buffer before 330s kill)
    return max(0.0, 320.0 - elapsed)


def _expected_improvement(mu, sigma, f_best, xi=0.01):
    """Compute expected improvement with a small exploitation term."""
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _ucb(mu, sigma, beta=2.0):
    """Upper confidence bound (minimization: lower is better, so we use mu - beta*sigma)."""
    return mu - beta * sigma


def _probability_of_improvement(mu, sigma, f_best, xi=0.01):
    """Probability of improvement."""
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu - xi) / sigma_safe
    return norm.cdf(z)


def _kmeans_select(points, n_clusters=5, max_iter=10):
    """Simple k-means to find representative points. Returns cluster centers."""
    if len(points) <= n_clusters:
        return points
    
    # Initialize with farthest-point sampling
    rng = np.random.RandomState(42)
    centers = np.empty((n_clusters, points.shape[1]))
    centers[0] = points[rng.randint(len(points))]
    
    for i in range(1, n_clusters):
        dists = np.min(
            np.sum((points[:, None, :] - centers[None, i-1:i, :])**2, axis=2),
            axis=1
        )
        # Actually, let's do proper farthest point
        min_dists = np.sum((points - centers[0:i]**2), axis=1)
        # Hmm, let me redo this properly
        pass
    
    # Simpler: just use k-means++ initialization
    centers = np.empty((n_clusters, points.shape[1]))
    centers[0] = points[rng.randint(len(points))]
    
    for i in range(1, n_clusters):
        dists = np.sum((points - centers[0:i])**2, axis=1)
        min_dists = np.min(dists, axis=1)
        probs = min_dists / (min_dists.sum() + 1e-10)
        centers[i] = points[rng.choice(len(points), p=probs)]
    
    # K-means iterations
    for _ in range(max_iter):
        # Assign points to nearest center
        dists = np.sum((points[:, None, :] - centers[None, :, :])**2, axis=2)
        assignments = np.argmin(dists, axis=1)
        
        # Update centers
        new_centers = np.zeros_like(centers)
        for j in range(n_clusters):
            mask = assignments == j
            if np.sum(mask) > 0:
                new_centers[j] = points[mask].mean(axis=0)
            else:
                new_centers[j] = centers[j]
        centers = new_centers
    
    return centers


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed function value
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        # Return a simple fallback
        return np.zeros_like(mu)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Check time budget
    remaining = _time_remaining()
    if remaining < 5.0:
        # Not enough time, return simple EI/cost
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / cost_safe
    
    # Compute multiple acquisition components
    xi = 0.01
    
    # 1. Expected Improvement
    z_ei = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    
    # 2. UCB (for minimization)
    ucb = mu - 2.0 * sigma_safe
    
    # 3. Probability of Improvement
    z_pi = (f_best - mu - xi) / sigma_safe
    pi = norm.cdf(z_pi)
    
    # 4. Cost-normalized versions
    ei_cost = ei / cost_safe
    ucb_cost = ucb / cost_safe  # Lower is better, so we want more negative
    pi_cost = pi / cost_safe
    
    # Combine: weighted sum with emphasis on EI and UCB
    # Use a blend that balances exploitation (EI, UCB) and exploration (sigma)
    scores = 0.5 * ei_cost + 0.3 * (-ucb_cost) + 0.2 * pi_cost
    
    # Add a small exploration bonus for high-uncertainty points
    sigma_norm = sigma_safe / (sigma_safe.max() + 1e-10)
    scores += 0.1 * sigma_norm / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
