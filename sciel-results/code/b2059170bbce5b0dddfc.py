"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for 2000 candidates.
    
    Uses a hybrid of Expected Improvement (EI) per unit cost and
    Thompson Sampling to balance exploration and exploitation.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more desirable to evaluate next.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Guard against zero or negative cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    ei_cost = ei / cost_safe
    
    # Thompson Sampling: sample from posterior and compute per-unit-cost
    # Use a deterministic pseudo-random sample for reproducibility within a call
    # We use a fixed seed based on the index to make it deterministic per candidate
    # Actually, for Thompson Sampling we want randomness. But we need to return
    # 2000 scores. Let's use a single random draw for the whole batch.
    rng = np.random.RandomState(42)
    z_sample = rng.standard_normal(size=mu.shape)
    ts_sample = mu - sigma_safe * z_sample  # Sampled function values
    ts_cost = -ts_sample / cost_safe  # Higher is better (lower objective / cost)
    
    # Blend EI and Thompson Sampling
    # Weight toward EI for exploitation, TS for exploration
    # Use a dynamic blend based on uncertainty
    # When sigma is large relative to improvement, favor exploration (TS)
    # When sigma is small, favor exploitation (EI)
    
    # Normalize both to comparable scales
    ei_cost_max = np.max(ei_cost) if np.max(ei_cost) > 1e-12 else 1.0
    ts_cost_max = np.max(ts_cost) if np.max(ts_cost) > 1e-12 else 1.0
    ei_cost_min = np.min(ei_cost)
    ts_cost_min = np.min(ts_cost)
    
    # Normalize to [0, 1]
    ei_norm = (ei_cost - ei_cost_min) / (ei_cost_max - ei_cost_min + 1e-12)
    ts_norm = (ts_cost - ts_cost_min) / (ts_cost_max - ts_cost_min + 1e-12)
    
    # Blend: 50% EI, 50% TS
    scores = 0.5 * ei_norm + 0.5 * ts_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
