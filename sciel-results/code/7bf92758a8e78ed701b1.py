"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _expected_improvement(mu, sigma, f_best):
    """Standard expected improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Clip to avoid negative values (shouldn't happen but just in case)
    return np.maximum(ei, 0.0)


def _ucb(mu, sigma, beta):
    """Upper confidence bound for minimization (negative of standard UCB)."""
    return mu - beta * sigma


def _soft_min(values, temperature=1.0):
    """Soft minimum approximation using log-sum-exp trick."""
    values = np.asarray(values, dtype=np.float64)
    min_val = np.min(values)
    # Soft min: -log-sum-exp(-values/temperature) * temperature
    # For numerical stability, subtract the min first
    shifted = -values / temperature
    shifted -= np.max(shifted)
    lse = np.log(np.sum(np.exp(shifted))) + np.max(shifted)
    return -lse * temperature


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending EI and UCB with adaptive weighting.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-10)
    
    n = len(mu)
    
    # Compute basic acquisition components
    ei = _expected_improvement(mu, sigma, f_best)
    ei_per_cost = ei / cost
    
    # Adaptive UCB-EI blend
    # Early: more exploration (UCB), Late: more exploitation (EI)
    # Use the current f_best relative to the range to determine phase
    # For styblinski tang, min is -78.3323, initial points are random in [-5,5]^2
    
    # Estimate progress based on how close f_best is to the known minimum
    # But we don't know the true minimum, so use a heuristic based on sigma magnitude
    sigma_mean = np.mean(sigma)
    sigma_min = np.min(sigma)
    
    # Adaptive beta for UCB: starts high, decreases as we learn more
    # Use a simple schedule based on the average sigma
    if sigma_mean > 1.0:
        beta = 2.0
    elif sigma_mean > 0.1:
        beta = 1.0
    else:
        beta = 0.5
    
    ucb = _ucb(mu, sigma, beta)
    ucb_per_cost = -ucb / cost  # Higher is better for minimization
    
    # Blend EI and UCB
    # Weight increases for EI as sigma decreases (more exploitation later)
    # Use a smooth transition
    blend_weight = np.clip(1.0 / (1.0 + sigma_mean), 0.3, 0.7)
    
    # Combined score
    scores = blend_weight * ei_per_cost + (1.0 - blend_weight) * ucb_per_cost
    
    # Check deadline
    if time.time() > deadline:
        # Return best available result
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: local refinement around promising points
    # Find top candidates and try to polish
    top_k = min(100, n)
    top_indices = np.argsort(scores)[-top_k:]
    
    if time.time() < deadline - 30:
        # Try a simple local search refinement
        # For each top candidate, consider small perturbations
        # But since we don't have coordinates, we can't do this directly
        # Instead, we can use the GP to suggest better points
        # This is a placeholder for potential refinement
        
        pass
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final check
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
