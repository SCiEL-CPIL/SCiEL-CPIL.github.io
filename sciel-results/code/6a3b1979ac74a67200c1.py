import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0 # Leave 50s for cleanup/other overhead

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if (time.time() - _START_TIME) > _TIME_LIMIT:
        # Return a safe, fast score. e.g. just mu (exploit current best)
        return mu.copy()

    # UCB for minimization
    # beta: exploration constant. 
    # Start with a moderate beta. 
    # We can estimate n_evals from cost. 
    # Initial cost is 5 points * ~1.5 = 7.5. 
    # Current total cost is passed? No, `cost` is the cost of the *candidates* (2000,).
    # Wait, the signature is `acquisition(mu, sigma, f_best, cost)`.
    # `cost` here is the cost of the 2000 candidates, not the total budget spent.
    # The prompt says: "mu, sigma and cost have shape (2000,)."
    # So we don't know the total budget spent directly from arguments.
    # However, we can assume the number of evaluations is roughly `total_cost_spent`.
    # But we don't have `total_cost_spent` in the args.
    # We can track it globally?
    # The evaluator calls `acquisition` in a loop.
    # We can increment a counter every time `acquisition` is called.
    
    global _CALL_COUNT
    _CALL_COUNT += 1
    
    # Heuristic: 
    # If we've done many calls, we are later in the optimization.
    # Reduce exploration.
    # Initial calls: 1. 
    # Budget 30. Avg cost ~1.5. So ~20 evaluations max.
    # So _CALL_COUNT will be around 20.
    
    # Let's use a fixed beta or a slowly decaying one.
    # beta = 2.0 is a common start.
    # Let's try beta = 2.0 * (1.0 / (1.0 + 0.1 * _CALL_COUNT))
    
    beta = 2.0 / (1.0 + 0.05 * _CALL_COUNT)
    
    # UCB score (higher is better for selection, so we minimize mu - beta*sigma)
    # Wait, standard UCB for minimization is: minimize mu - sqrt(beta)*sigma.
    # So acquisition score to MAXIMIZE is: -mu + sqrt(beta)*sigma ? 
    # Or just compute `ucb = mu - sqrt(beta)*sigma` and return `-ucb`?
    # No, the prompt says "Return 2000 finite scores (higher = evaluate next)".
    # So if we want to pick the point with LOWEST `mu - sqrt(beta)*sigma`, 
    # we should return `-(mu - sqrt(beta)*sigma)` = `-mu + sqrt(beta)*sigma`.
    
    ucb = mu - np.sqrt(beta) * sigma
    
    # Cost Penalty
    # We want to prefer lower cost.
    # Cost is in [1.0, 2.0].
    # Let's use a multiplicative penalty that is 1.0 at cost=1.5 and drops off.
    # Or additive: score = -ucb - lambda * cost.
    # If we use multiplicative: score = (-ucb) * penalty.
    # If -ucb is negative (which it likely is since mu is negative and sigma is small), 
    # multiplying by a penalty < 1 makes it MORE negative, which is WORSE for maximization.
    # So multiplicative penalty on a negative score is tricky.
    
    # Let's stick to Additive Cost Penalty on the "Minimization Score".
    # MinScore = mu - sqrt(beta)*sigma + lambda * cost.
    # We want to MINIMIZE MinScore.
    # So Acquisition Score (to MAXIMIZE) = -MinScore = -mu + sqrt(beta)*sigma - lambda * cost.
    
    # Choose lambda.
    # Cost range [1, 2]. Difference is 1.
    # Function values range ~ 4.
    # If lambda = 0.5, cost difference of 1 changes score by 0.5.
    # This is significant but not dominant.
    
    lambda_cost = 0.5
    
    score = -ucb - lambda_cost * cost
    
    # Normalize? 
    # The scale of `score` might vary. 
    # But relative ordering is what matters.
    
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score

_CALL_COUNT = 0
