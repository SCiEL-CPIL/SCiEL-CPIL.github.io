import numpy as np
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard
    if time.time() - _start_time > 328.0:
        # Return a safe fallback: just exploit the mean
        scores = f_best - mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Adaptive exploration weight
    # When f_best is large (poor), we need high exploration.
    # When f_best is small (good), we need low exploration (exploitation).
    # f_best is the minimum observed.
    # Scale f_best to [0, 1] roughly?
    # Powell 4d values can be large.
    # Let's use a sigmoid-like mapping or simple division.
    
    # If f_best is 0, beta should be small.
    # If f_best is 100, beta should be large.
    
    # beta = 1.0 + 2.0 * (f_best / (1.0 + f_best))
    # This saturates at 3.0 for large f_best.
    
    beta = 1.0 + 2.0 * (f_best / (1.0 + f_best))
    
    # Cost penalty
    # Cost is in [1, 2].
    # We subtract a penalty proportional to cost.
    # Center the cost at 1.5 to avoid biasing the baseline score.
    # lambda controls the strength of cost awareness.
    # Since the budget is tight (30), cost matters.
    # But the function value matters more.
    # Let's use a small lambda.
    lambda_cost = 0.1
    
    # Calculate scores
    # Maximize: f_best - mu + beta * sigma - lambda * (cost - 1.5)
    ucb_term = f_best - mu + beta * sigma
    cost_penalty = lambda_cost * (cost - 1.5)
    
    scores = ucb_term - cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
