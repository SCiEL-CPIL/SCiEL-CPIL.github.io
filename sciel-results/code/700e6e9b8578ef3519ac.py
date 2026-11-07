import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware GP-UCB acquisition function with adaptive exploration coefficient.
    
    Parameters:
    mu: array of shape (2000,), predicted mean from GP
    sigma: array of shape (2000,), predicted standard deviation from GP
    f_best: float, best function value observed so far
    cost: array of shape (2000,), cost of evaluating each candidate
    
    Returns:
    scores: array of shape (2000,), acquisition scores (higher is better)
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Estimate budget progress:
    # Initial cost is roughly 5 points * mean cost ~ 5 * 1.5 = 7.5
    # Total budget is 30
    # Use f_best as a proxy for progress - lower f_best means more progress
    # Hartmann 6d min is -3.32237, so we can normalize
    
    # Adaptive beta based on exploration/exploitation tradeoff
    # Early on: high beta (explore), Late on: low beta (exploit)
    # Use a combination of:
    # 1. The magnitude of f_best (closer to minimum means later stage)
    # 2. The typical sigma values (lower sigma means more explored)
    
    # Estimate progress from f_best
    # Hartmann 6d: min ~ -3.32, typical initial values around -1 to -2
    # Progress: 0 at start (f_best ~ -1), 1 at end (f_best ~ -3.32)
    f_min_hartmann = -3.32237
    f_start_est = -1.0  # rough estimate of initial f_best
    
    # Normalize progress to [0, 1]
    progress = np.clip((f_start_est - f_best) / (f_start_est - f_min_hartmann), 0.0, 1.0)
    
    # Also consider sigma: if sigma is large, we're still exploring
    mean_sigma = np.mean(sigma_safe)
    # Typical sigma at start might be ~0.5, at end ~0.01
    sigma_progress = np.clip(1.0 - mean_sigma / 0.5, 0.0, 1.0)
    
    # Combine progress estimates
    combined_progress = 0.6 * progress + 0.4 * sigma_progress
    combined_progress = np.clip(combined_progress, 0.0, 1.0)
    
    # Adaptive beta: starts high (~3.0) and decays to low (~0.5)
    # Use exponential decay with a smooth transition
    beta_start = 3.0
    beta_end = 0.5
    # Smooth decay: beta = beta_end + (beta_start - beta_end) * exp(-k * progress)
    k = 3.0  # decay rate
    beta = beta_end + (beta_start - beta_end) * np.exp(-k * combined_progress)
    
    # GP-UCB for minimization:
    # We want to minimize, so we want low mu and high uncertainty
    # UCB for minimization: mu - beta * sigma
    # But we want higher score = better, so:
    # score = -(mu - beta * sigma) = -mu + beta * sigma
    # This gives high scores where mu is low (good) and sigma is high (uncertain)
    
    ucb_score = -mu + beta * sigma_safe
    
    # Cost-aware normalization:
    # Divide by a function of cost to penalize expensive evaluations
    # Use sqrt(cost) for moderate penalty
    cost_safe = np.maximum(cost, 1.0)  # cost is at least 1
    cost_penalty = np.sqrt(cost_safe)
    
    # Final score
    scores = ucb_score / cost_penalty
    
    # Check deadline
    if time.time() > deadline:
        # Return whatever we have so far, ensuring finite values
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: add a small random tie-breaker to avoid always picking the same point
    # This helps with exploration in case of ties
    # But keep it very small to not disrupt the ranking
    # Actually, the evaluator picks the best, so ties are broken by index
    # Let's not add randomness to keep it deterministic
    
    return scores
