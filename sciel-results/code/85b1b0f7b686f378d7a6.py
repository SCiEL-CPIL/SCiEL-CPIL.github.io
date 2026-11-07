"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = None
_DEADLINE = 310.0  # 330s limit, leave 20s margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - lowest observed value
    cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    global _start_time
    
    # Initialize start time if not set
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're too close to the limit, return a quick fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: simple UCB-like score
        sigma_safe = np.maximum(sigma, 1e-8)
        fallback = (f_best - mu - 1.0 * sigma_safe) / np.maximum(cost, 1e-8)
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safe sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Safe cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-8)
    
    # Cost-aware EI: EI per unit cost
    ei_cost = ei / cost_safe
    
    # Upper Confidence Bound (UCB) term - more aggressive exploration
    # Use a dynamic exploration parameter that decreases over time
    # This starts with higher exploration and converges to exploitation
    elapsed_fraction = min(elapsed / _DEADLINE, 1.0)
    beta = 0.5 + 0.5 * (1.0 - elapsed_fraction)  # Beta ranges from 1.0 to 0.5
    
    ucb = (f_best - mu - beta * sigma_safe) / cost_safe
    
    # Additional uncertainty-weighted improvement that emphasizes high sigma regions
    # This captures the idea that high uncertainty areas may have better optima
    # Use a scaled version of (f_best - mu) weighted by sigma to find promising uncertain regions
    uncertainty_gain = (f_best - mu) * np.log1p(sigma_safe / 0.1) / cost_safe
    
    # Normalize each component to comparable scales
    def z_normalize(x):
        mean = np.mean(x)
        std = np.std(x) + 1e-10
        return (x - mean) / std
    
    ei_cost_norm = z_normalize(ei_cost)
    ucb_norm = z_normalize(ucb)
    uncertainty_norm = z_normalize(uncertainty_gain)
    
    # Combine with adaptive weights
    # More weight to EI later, more weight to uncertainty early
    ei_weight = 0.5 + 0.3 * elapsed_fraction  # 0.5 to 0.8
    ucb_weight = 0.3 - 0.1 * elapsed_fraction  # 0.3 to 0.2
    uncertainty_weight = 1.0 - ei_weight - ucb_weight  # remaining
    
    # Ensure weights are positive
    weights = np.array([max(ei_weight, 0.1), max(ucb_weight, 0.1), max(uncertainty_weight, 0.1)])
    weights = weights / np.sum(weights)
    
    scores = weights[0] * ei_cost_norm + weights[1] * ucb_norm + weights[2] * uncertainty_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
