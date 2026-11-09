"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Cache for timing guard
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4D.
    Uses a Thompson Sampling-inspired heuristic with adaptive exploration.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    global _start_time, _call_count
    
    # Initialize timing guard if not set
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: return early if we're close to the 330s limit
    elapsed = time.time() - _start_time
    if elapsed > 325:  # 5 second safety margin
        # Return a simple fallback: prioritize low cost and low predicted value
        return -mu / (cost + 1e-9)
    
    _call_count += 1
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Use Thompson Sampling: sample from posterior for each candidate
    # For minimization, we want to sample low values
    # Sample: f_sample = mu - sigma * z where z ~ N(0,1)
    # This gives us a stochastic estimate of the function value
    
    # Generate random samples for Thompson Sampling
    rng = np.random.default_rng()
    z_samples = rng.standard_normal(size=mu.shape)
    
    # Thompson sample: lower is better, so we sample mu - sigma * z
    # This means when z is positive, we get lower values (better for minimization)
    ts_samples = mu - sigma * z_samples
    
    # Expected Improvement for minimization using Thompson samples
    # EI = max(0, f_best - f_sample)
    ei_ts = np.maximum(f_best - ts_samples, 0.0)
    
    # Also compute deterministic EI for stability
    z_det = (f_best - mu) / sigma
    ei_det = (f_best - mu) * norm.cdf(z_det) + sigma * norm.pdf(z_det)
    ei_det = np.maximum(ei_det, 0.0)
    
    # Blend Thompson and deterministic EI
    # Use more Thompson sampling early on (high uncertainty), more deterministic later
    # Estimate progress by comparing f_best to typical scale
    # Shekel 4D min is -10.5364, initial values are around -1 to -5
    # Use a simple heuristic: if f_best is close to optimum, exploit more
    progress = np.clip((f_best - (-10.5364)) / 10.0, 0, 1)
    # progress = 1 means far from optimum (exploit less), 0 means near optimum (exploit more)
    # Actually, let's use call count as a proxy
    blend_ts = min(1.0, 0.3 + 0.7 * (1.0 - min(_call_count / 20.0, 1.0)))
    blend_det = 1.0 - blend_ts
    
    ei_combined = blend_ts * ei_ts + blend_det * ei_det
    
    # Cost-aware: divide by cost to get improvement per unit cost
    cost_aware = ei_combined / (cost + 1e-9)
    
    # Add exploration bonus based on uncertainty, scaled by progress
    # More exploration early, less later
    exploration_weight = 0.15 * (1.0 - min(_call_count / 30.0, 1.0))
    exploration_bonus = exploration_weight * sigma / (cost + 1e-9)
    
    # Combine
    scores = cost_aware + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
