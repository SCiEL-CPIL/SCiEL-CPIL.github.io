# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Dynamic, budget-aware Thompson Sampling acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Safety margin: ensure we return before the 300s limit
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute improvement: how much better than f_best
    improvement = f_best - mu
    
    # Thompson Sampling: sample from the posterior
    # Use a temperature parameter that adapts to remaining budget
    # Since we don't know the exact remaining budget, we use a heuristic:
    # - Early in the search (high uncertainty relative to improvement), explore more
    # - Later (low uncertainty, or improvement is large), exploit more
    
    # Temperature parameter: controls exploration vs exploitation
    # We use the ratio of sigma to improvement magnitude as a proxy for uncertainty
    # When improvement is large relative to sigma, we should exploit (lower temperature)
    # When improvement is small relative to sigma, we should explore (higher temperature)
    
    # Compute a local temperature for each point
    # Base temperature: 1.0 (standard Thompson sampling)
    # Adjust based on how "confident" we are in the improvement
    
    # For points with large positive improvement (likely good), lower temperature (more exploitation)
    # For points with small or negative improvement, higher temperature (more exploration)
    
    # Normalize improvement by sigma to get a z-score
    z = improvement / sigma_safe
    
    # Temperature: higher when z is small (uncertain), lower when z is large (confident)
    # Use a smooth function: temperature = 1 + alpha * (1 - sigmoid(z))
    # Where sigmoid(z) is close to 1 for large z (confident) and close to 0 for small z (uncertain)
    
    # Use a simple heuristic: temperature = 1 + max(0, 1 - z/2)
    # This gives temperature in [1, 2] range
    # When z >= 2, temperature = 1 (pure exploitation)
    # When z <= 0, temperature = 2 (more exploration)
    
    temperature = 1.0 + np.maximum(0.0, 1.0 - z / 2.0)
    
    # Sample from the posterior: y ~ N(mu, sigma^2 * temperature^2)
    # We want to maximize the sampled value (since lower is better, we minimize)
    # So we sample and then negate to convert to maximization
    
    # Use a fixed random seed for reproducibility within a single call
    # But different calls should use different samples
    # Since we can't maintain state, we use the current time as a seed
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    
    # Sample from N(mu, sigma * temperature)
    samples = mu + sigma_safe * temperature * rng.standard_normal(len(mu))
    
    # We want to find the point with the lowest sampled value (since lower is better)
    # So the score should be -samples (higher score = lower sampled value = better)
    
    # But we also need to account for cost
    # Divide by cost to get cost-aware score
    scores = -samples / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin: 20s before the 300s limit
        # Return the current scores as-is
        return scores
    
    return scores
# EVOLVE-BLOCK-END
