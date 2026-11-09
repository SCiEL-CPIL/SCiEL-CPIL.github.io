"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_call_count = 0
_TIME_LIMIT = 295  # Seconds - leave safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Thompson Sampling with PI.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP posterior mean in objective units
    sigma : array-like, shape (2000,)
        GP posterior std in objective units
    f_best : float
        Lowest observed objective value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    global _call_count
    _call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        return np.zeros_like(np.asarray(mu, dtype=np.float64))
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # --- Component 1: Probability of Improvement (PI) ---
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    pi_scores = norm.cdf(z)
    
    # --- Component 2: Thompson Sampling ---
    # Use multiple Monte Carlo samples to estimate expected minimum cost-normalized objective
    # This inherently balances exploration (via sampling) and exploitation (via minimum)
    n_samples = 10
    rng = np.random.default_rng(seed=42 + _call_count)
    
    # Sample from the GP posterior: y ~ N(mu, sigma^2)
    samples = rng.normal(mu, sigma, size=(n_samples, len(mu)))
    
    # Cost-normalized objective for each sample and candidate
    # We want to minimize objective/cost, so we look at the minimum across samples
    cost_normalized = samples / cost[None, :]
    
    # For each candidate, take the minimum across samples (optimistic view)
    # Then negate because we want to maximize (higher score = better)
    ts_scores = -np.min(cost_normalized, axis=0)
    
    # Normalize Thompson Sampling scores to [0, 1] range for blending
    ts_min = np.min(ts_scores)
    ts_max = np.max(ts_scores)
    ts_range = ts_max - ts_min + 1e-9
    ts_normalized = (ts_scores - ts_min) / ts_range
    
    # --- Blend PI and Thompson Sampling ---
    # PI is in [0, 1], TS is normalized to [0, 1]
    # Weight: PI provides stable probability-based signal, TS provides exploration
    w_pi = 0.5
    w_ts = 0.5
    
    blended = w_pi * pi_scores + w_ts * ts_normalized
    
    # --- Cost-aware normalization ---
    # Divide by cost to prefer cheaper evaluations
    # cost range is [1, 2], so this has modest effect
    scores = blended / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
