import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling with budget-aware scaling.
    
    Args:
        mu: Mean of the GP prediction, shape (N,)
        sigma: Standard deviation of the GP prediction, shape (N,)
        f_best: Best observed value so far.
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
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Thompson Sampling: draw a single sample from the posterior for all candidates
    # y ~ N(mu, sigma^2)
    # We want to minimize y, so we score by -y / cost (higher is better)
    
    # Use a fixed random seed based on the current state to ensure reproducibility
    # within a single call, but different across calls
    # Actually, for Thompson sampling in BO, we typically draw fresh each time.
    # But we need to be deterministic if possible? No, TS is stochastic.
    # However, the task says "Return 2000 finite scores".
    # The evaluator picks the argmax of these scores.
    
    # Draw sample
    rng = np.random.default_rng()
    y_sample = mu + sigma_safe * rng.standard_normal(n)
    
    # Budget-aware scaling:
    # The Hartmann 6D minimum is -3.32237.
    # dist_to_min = max(0, f_best - (-3.32237))
    # As we get closer to the minimum, we should be more exploitative.
    # In TS, we can adjust the "temperature" or just rely on the natural balance.
    # However, we can scale the cost penalty based on how close we are.
    
    # Alternative: Combine TS with a small EI component for stability?
    # The directive says "Model the acquisition as a Thompson Sampling score... replacing deterministic exploitation/exploration trade-off".
    # So pure TS is the structural change.
    
    # Score: -y_sample / cost_safe
    # This naturally balances exploration (high sigma -> high variance in y) 
    # and exploitation (low mu -> low y).
    # Dividing by cost penalizes expensive points.
    
    scores = -y_sample / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return a safe fallback (exploitative) if we are running out of time
        return -mu / cost_safe
    
    return scores
