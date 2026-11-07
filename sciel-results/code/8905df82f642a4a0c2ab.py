import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margin for time
    # The computation is trivial, so time check is mostly for compliance with directive
    if time.time() - start_time > 290:
        # Return a fallback, e.g., based on cost only (cheapest)
        # But this shouldn't happen
        return 1.0 / np.maximum(cost, 1e-9)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB for minimization: we want low mu, high sigma
    # Score = (mu + beta * sigma) / cost
    # Wait, for minimization, we want to MINIMIZE mu.
    # UCB is usually for maximization: mu + beta * sigma.
    # For minimization, we want to MINIMIZE mu.
    # So we should use Lower Confidence Bound (LCB)?
    # LCB = mu - beta * sigma.
    # We want to select the point with the LOWEST LCB?
    # No, the acquisition function returns scores where HIGHER is better.
    # So we want to MAXIMIZE the score.
    # If we want to find low values, we should maximize the "potential for low values".
    # The potential for low values is higher when mu is low AND sigma is high.
    # So we want to maximize (-mu + beta * sigma)? Or minimize (mu - beta * sigma)?
    # Let's think.
    # If mu is low, it's good.
    # If sigma is high, it's good (uncertainty).
    # So a score of (-mu + beta * sigma) would be high for low mu and high sigma.
    # Let's check:
    # Point A: mu=0, sigma=1. Score = 0 + beta = beta.
    # Point B: mu=10, sigma=1. Score = -10 + beta.
    # Point A is better. Correct.
    # Point C: mu=0, sigma=10. Score = 0 + 10*beta.
    # Point D: mu=-1, sigma=1. Score = 1 + beta.
    # If beta=1, C=10, D=2. C is preferred.
    # Is C better? It has higher uncertainty. Yes, for exploration.
    
    # So Score = (-mu + beta * sigma) / cost.
    # Or Score = (beta * sigma - mu) / cost.
    
    beta = 2.0
    score = (beta * sigma_safe - mu) / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
