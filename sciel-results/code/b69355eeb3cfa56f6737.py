import numpy as np
import time
from scipy.stats import norm

# Global state for tracking the optimization step within a seed
_state = {
    "call_count": 0,
    "prev_f_best": None,
    "start_time": None,
    "total_bo_steps": 25  # 30 total evaluations - 5 initial = 25 BO steps
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses a Thompson Sampling-inspired strategy that samples from the posterior
    distribution for each candidate, providing a natural balance between
    exploration and exploitation without manual tuning of exploration constants.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (N,)
    sigma (np.ndarray): Std dev of the GP posterior, shape (N,)
    f_best (float): Best observed value so far (minimum).
    
    Returns:
    np.ndarray: Scores for each candidate, higher is better.
    """
    global _state
    
    # Initialize start time if not set
    if _state["start_time"] is None:
        _state["start_time"] = time.time()
    
    # Check deadline guard: if we are close to the limit, return a safe fallback
    elapsed = time.time() - _state["start_time"]
    if elapsed > 325.0:
        mu = np.asarray(mu, dtype=np.float64)
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Detect if this is a new seed (new optimization run)
    is_new_seed = False
    if _state["prev_f_best"] is not None:
        if f_best > _state["prev_f_best"] + 1e-9 or _state["call_count"] >= _state["total_bo_steps"]:
            is_new_seed = True
            
    if is_new_seed:
        _state["call_count"] = 0
    else:
        _state["call_count"] += 1
        
    _state["prev_f_best"] = f_best
    
    # Ensure step count is within valid range [0, 24]
    current_step = min(_state["call_count"], _state["total_bo_steps"] - 1)
    
    # Calculate progress in the optimization loop [0, 1]
    progress = current_step / max(_state["total_bo_steps"] - 1, 1)
    
    # Ensure inputs are clean
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle any non-finite values in sigma or mu
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(sigma), mu, 0.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-12)
    
    # Thompson Sampling: sample from the posterior Gaussian for each candidate
    # The sampled value represents a plausible function value at that point
    # We want to find the candidate with the minimum sampled value (for minimization)
    # So we return negative of the sampled value as the score (higher is better)
    
    # Generate random normal samples for Thompson Sampling
    rng = np.random.default_rng()
    eps = rng.standard_normal(size=mu.shape)
    
    # Sample from the posterior: y = mu + sigma * eps
    sampled_values = mu + sigma * eps
    
    # For minimization, we want the minimum sampled value
    # Score = -sampled_value (so higher score = lower sampled value = better)
    scores = -sampled_values
    
    # Add a small exploitation bonus to break ties and prefer points near f_best
    # This helps when Thompson Sampling gives similar scores
    # The bonus is small and decays over time
    if progress < 0.8:  # Only in first 80% of steps
        # Proximity bonus: points with mu close to f_best get a small boost
        proximity = np.maximum(f_best - mu, 0.0)
        bonus_weight = 0.1 * (1.0 - progress / 0.8)
        scores = scores + bonus_weight * proximity
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
