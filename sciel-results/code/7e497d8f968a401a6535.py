"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition for rosenbrock 2d.
    
    Strategy: 
    - Use Thompson Sampling: sample from posterior mu + sigma * N(0,1)
    - Normalize by cost to get expected value per unit cost
    - Add a small deterministic exploitation term based on current best
    - Use a time guard to ensure we return within budget
    """
    # Time guard: track start time
    t_start = time.time()
    deadline = t_start + 250.0  # Safety margin before 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if mu.size == 0:
        return np.array([])
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: sample noisy objective values
    # Use a fixed seed for reproducibility within a call, but different across calls
    # Actually, we should use random sampling for proper Thompson Sampling
    rng = np.random.default_rng()
    
    # Primary Thompson Sampling score
    # Sample from the posterior: mu + sigma * z where z ~ N(0,1)
    z = rng.standard_normal(mu.shape)
    ts_sample = mu + sigma * z
    
    # Cost-normalized Thompson Sampling score (higher is better for minimization)
    # We want to minimize, so lower ts_sample/cost is better
    # Acquisition score should be higher for more promising points
    # So we negate: -ts_sample / cost
    ts_score = -ts_sample / cost
    
    # Add a deterministic exploitation component
    # Points with low mu are promising; points near f_best are safe
    # Use a weighted combination
    # Exploitation: favor low mu, penalize high cost
    exploit_score = -mu / cost
    
    # Exploration bonus: favor uncertain regions (high sigma) but balanced
    # Use PI-like bounded term: sigma / (sigma + |mu - f_best|)
    # This is in [0, 1] and favors uncertain, promising regions
    diff = np.abs(mu - f_best)
    pi_term = sigma / (sigma + diff + 1e-9)
    
    # Combine: primary TS score with a small exploitation boost
    # The TS score already captures both exploration and exploitation stochastically
    # Add a small deterministic component to stabilize
    alpha = 0.8  # Weight for Thompson Sampling
    beta = 0.2   # Weight for deterministic exploitation
    
    score = alpha * ts_score + beta * exploit_score
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -1e10)
    
    # Check time budget
    if time.time() > deadline:
        # Return immediately with whatever we have
        return score
    
    # Optional: refine with a second pass if time permits
    elapsed = time.time() - t_start
    if elapsed < 200.0:  # If we have plenty of time left
        # Do a refined Thompson Sampling with multiple samples
        n_samples = 5
        ts_samples = mu[None, :] + sigma[None, :] * rng.standard_normal((n_samples, mu.shape[0]))
        ts_scores = -ts_samples / cost[None, :]
        # Use the mean of multiple TS samples for stability
        ts_score_refined = np.mean(ts_scores, axis=0)
        
        # Blend with original
        score = 0.5 * ts_score + 0.5 * ts_score_refined
        score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
