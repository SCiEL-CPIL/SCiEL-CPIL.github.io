"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline tracking
_start_time = None
_DEADLINE = 325.0  # Safety margin before 330s kill

# Shekel 4D minimum is approx -10.5364
SHEKEL_MIN = -10.5364

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization using Thompson Sampling.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs (1 + mean(x scaled to [0,1]))
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a safe fallback: just use -mu/cost to avoid any further computation
        return -mu / (cost + 1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling:
    # 1. Sample a single objective value from the GP posterior for each candidate.
    #    y_sample = mu + sigma * z, where z ~ N(0, 1)
    #    Since we want to minimize, we want low y_sample.
    #    However, the acquisition function should return higher scores for better candidates.
    #    So we can return -y_sample.
    
    # 2. Cost-awareness:
    #    The task states: "Evaluating x costs 1 + mean(x scaled to [0, 1])".
    #    We have a budget of 30 total cost.
    #    Standard Thompson Sampling doesn't inherently account for varying costs per evaluation.
    #    A common heuristic is to penalize high-cost evaluations.
    #    One approach: Score = -y_sample / cost.
    #    This rewards low predicted objective values and low costs.
    #    However, dividing by cost can be noisy if cost is small.
    #    Another approach: Score = -y_sample * (1/cost).
    #    Let's try -y_sample / cost.
    
    # Generate standard normal noise
    z = np.random.randn(mu.shape[0])
    
    # Sample objective values
    y_sample = mu + sigma * z
    
    # For minimization, lower y_sample is better.
    # We want higher acquisition scores.
    # So base_score = -y_sample.
    
    # Cost adjustment:
    # If cost is high, we should be less inclined to evaluate, so divide by cost.
    # score = -y_sample / cost
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    score = -y_sample / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
