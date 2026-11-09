import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on hartmann 6d.
    Uses Thompson Sampling: draw a single posterior sample y ~ N(mu, sigma^2)
    for all candidates, then score by -y / cost (higher = evaluate next).
    
    Args:
        mu: Mean of the GP prediction, shape (N,)
        sigma: Standard deviation of the GP prediction, shape (N,)
        f_best: Best observed value so far (scalar).
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,), higher = evaluate next
    """
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    # Input validation and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma_safe = np.maximum(sigma_safe, 1e-8)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost_safe = np.where(np.isfinite(cost), cost, 1.0)
    cost_safe = np.maximum(cost_safe, 1e-8)
    
    # Thompson Sampling: draw a single sample from the posterior
    # y ~ N(mu, sigma^2) for all candidates
    # Use a fixed seed per call? No - we want randomness for exploration.
    # But we need reproducibility within a single acquisition call.
    # Draw one sample for all 2000 candidates.
    
    # For better exploitation/exploitation balance, we can use a slightly
    # shrunk sigma to bias toward exploitation when f_best is good,
    # or keep full sigma for pure Thompson Sampling.
    
    # Pure Thompson Sampling: score = -y / cost
    # y = mu + sigma * z, where z ~ N(0, 1)
    
    # Draw random numbers - use numpy's default RNG (no fixed seed for diversity)
    z = np.random.standard_normal(n)
    
    # Posterior sample
    y = mu + sigma_safe * z
    
    # Score: we want to minimize y, penalize by cost
    # Higher score = better to evaluate
    # score = -y / cost
    scores = -y / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return a safe fallback
        return np.zeros(n, dtype=np.float64)
    
    return scores
