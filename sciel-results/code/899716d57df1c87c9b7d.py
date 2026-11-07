"""Cost-aware acquisition for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We track time to ensure we don't exceed the 300s limit.
# Since this function is called many times, we use a lightweight global state
# to check if we are approaching the deadline. If we are, we return a simple,
# fast heuristic to avoid overhead and ensure we finish in time.

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds before the 300s limit to start bailing out

def _is_deadline_near():
    """Check if we are close to the wall-clock deadline."""
    return (time.time() - _start_time) > (300.0 - _DEADLINE_MARGIN)

def _zscore(arr):
    """
    Robust z-scoring. If std is very small, return arr - mean to avoid division by zero.
    """
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-8:
        return arr - mean
    return (arr - mean) / std

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidates using a hybrid approach:
    1. Thompson Sampling (TS) for exploration/exploitation balance.
    2. Cost-aware scaling.
    3. Fallback to simple EI/cost if deadline is near.
    
    Parameters:
    -----------
    mu : np.ndarray
        Mean of the GP posterior, shape (2000,).
    sigma : np.ndarray
        Standard deviation of the GP posterior, shape (2000,).
    f_best : float
        Best observed value so far.
    cost : np.ndarray
        Cost of evaluating each candidate, shape (2000,).
        
    Returns:
    --------
    np.ndarray
        Acquisition scores, shape (2000,). Higher is better.
    """
    # Deadline guard: if we are close to the limit, use a very fast heuristic
    if _is_deadline_near():
        # Simple EI / cost, no z-scoring, no TS
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, -np.inf)

    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8) # Safety for cost division

    # --- Structural Change: Thompson Sampling + Cost Awareness ---
    # Instead of deterministic EI/UCB combination, we sample from the posterior.
    # This naturally balances exploration and exploitation.
    
    # Sample from the GP posterior
    z_samples = np.random.randn(len(mu))
    y_samples = mu + sigma * z_samples
    
    # Calculate improvement for each sample
    # Since we are minimizing, improvement is f_best - y_sample
    improvements = f_best - y_samples
    
    # Cost-aware score: Improvement per unit cost
    # This is the core of the Thompson Sampling acquisition
    scores = improvements / cost
    
    # --- Refinement: Add a small deterministic EI component for stability ---
    # Pure TS can be noisy. We blend it with a deterministic EI/cost signal.
    # This helps in the early stages or when sigma is very small.
    
    z_ei = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    ei_cost = ei / cost
    
    # Z-score both signals to make them comparable
    z_ts = _zscore(scores)
    z_ei_cost = _zscore(ei_cost)
    
    # Combine: 80% Thompson Sampling, 20% Deterministic EI
    # This leverages the exploration of TS while stabilizing with EI.
    final_scores = 0.8 * z_ts + 0.2 * z_ei_cost
    
    # Ensure all values are finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, -np.inf)
    
    return final_scores
# EVOLVE-BLOCK-END
