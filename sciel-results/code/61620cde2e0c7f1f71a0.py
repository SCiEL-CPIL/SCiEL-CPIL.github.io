import numpy as np
import time

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
    Uses a quantile-adaptive hybrid acquisition that blends exploration and
    exploitation based on the actual posterior uncertainty landscape.
    
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
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Structural change: quantile-adaptive exploration
    # Compute a reference uncertainty level from the top candidates
    # This makes beta adapt to the actual posterior landscape rather than a fixed schedule
    
    # Use the 90th percentile of sigma as a reference for "high uncertainty"
    sigma_ref = np.percentile(sigma, 90)
    if sigma_ref <= 1e-12:
        sigma_ref = 1.0
    
    # Normalize sigma to [0, 1] range for stable computation
    sigma_norm = sigma / sigma_ref
    
    # Adaptive exploration weight: starts high, decays, but has a floor
    # The floor ensures we never fully stop exploring
    beta_base = 1.5 + (0.3 - 1.5) * progress  # Linear from 1.5 to 0.3
    beta_floor = 0.15
    
    # Quantile-based adjustment: if the top-scoring candidates have high uncertainty,
    # increase exploration; if they're certain, decrease it
    # Use the mean sigma_norm of the top 10% by a preliminary score
    prelim_score = beta_base * sigma_norm - mu
    top_idx = np.argsort(prelim_score)[-max(1, len(prelim_score) // 10):]
    top_sigma_mean = np.mean(sigma_norm[top_idx]) if len(top_idx) > 0 else 0.5
    
    # Adjust beta: if top candidates are uncertain, boost exploration
    # If top candidates are certain, allow more exploitation
    beta_adjustment = 0.5 * (1.0 - top_sigma_mean)  # Range [0, 0.5]
    beta = max(beta_floor, beta_base + beta_adjustment)
    
    # Hybrid score: blend UCB with a variance-weighted exploitation term
    # score = beta * sigma - mu + gamma * (sigma / (sigma + epsilon)) * (f_best - mu)
    # The second term rewards points that are both uncertain AND close to f_best
    epsilon = 1e-8
    gamma = 0.3 * (1.0 - progress)  # Decays over time
    
    # Variance-weighted proximity term: high when sigma is large and mu is close to f_best
    proximity = (sigma / (sigma + epsilon)) * np.maximum(f_best - mu, 0.0)
    
    scores = beta * sigma - mu + gamma * proximity
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
