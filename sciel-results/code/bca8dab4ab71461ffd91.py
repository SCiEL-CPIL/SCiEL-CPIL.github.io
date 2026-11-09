"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    # Safety margin: return simple EI if we are dangerously close to the 330s limit
    if elapsed > 325.0:
        z = (f_best - mu) / (sigma + 1e-9)
        from scipy.stats import norm
        return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost

    # Dynamic exploration coefficient.
    # Starts high to encourage exploration, shrinks as remaining budget decreases.
    # Total budget is 30. Initial cost is approx 1 + 0.5 = 1.5 * 5 = 7.5.
    # Remaining cost = 30 - current_cost.
    # We want beta to be proportional to sqrt(remaining budget / total budget) or similar.
    # Let's use a normalized remaining budget factor.
    current_cost = np.mean(cost)  # cost is the cumulative cost vector, likely all same value? 
                                  # "cost have shape (2000,)". In the loop, cost is passed. 
                                  # Usually cost[i] is the cumulative cost spent so far, or the cost of candidate i?
                                  # "Evaluating x costs 1 + mean(x scaled to [0, 1]). Runs start with 5 uniform random points (cost counted). While total cost is below 30..."
                                  # The signature is acquisition(mu, sigma, f_best, cost). 
                                  # In standard BO loops, `cost` often represents the cumulative cost incurred so far, 
                                  # or the cost associated with each candidate. 
                                  # Given "cost have shape (2000,)", and we return 2000 scores, 
                                  # it is highly likely `cost[i]` is the cost to evaluate candidate i, 
                                  # OR the cumulative cost is the same for all? 
                                  # If it's the cost to evaluate candidate i, then cost[i] = 1 + mean(x_i scaled).
                                  # If it's cumulative, it would be a scalar or repeated. 
                                  # Let's assume cost[i] is the marginal cost of evaluating candidate i.
                                  # To get remaining budget, we need total spent. 
                                  # We don't have total spent directly. 
                                  # However, we can estimate remaining budget if we assume the loop stops at 30.
                                  # Let's look at the seed: `return ... / cost`. This implies cost is the marginal cost of the candidate.
                                  # So `cost[i]` is the cost of candidate i.
                                  # We need total cost spent so far. We don't have it.
                                  # But we can approximate the "stage" of the search.
                                  # Alternatively, just use a fixed beta that works well, or decay based on number of iterations? 
                                  # We don't have iteration count.
                                  # Let's stick to a strong exploitation bias + UCB with a moderate beta.
                                  # The inspiration mentioned adding -0.1*mu to Cost-Aware EI improved score.
                                  # Let's try UCB: score = -mu + beta * sigma / cost.
                                  # Why / cost? To normalize by evaluation cost.
                                  # Beta: Let's use 1.0 as a starting point, maybe scaled.
    
    # UCB with cost normalization
    # Higher score = better.
    # We want to minimize f. So -mu is good.
    # sigma is uncertainty.
    # We divide by cost to penalize expensive points.
    
    beta = 1.5  # Exploration weight
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Cost-aware UCB
    # The term sigma/cost encourages exploration in cheap, uncertain regions.
    # -mu encourages exploitation of low mean regions.
    score = -mu + beta * (sigma_safe / cost)
    
    # Add a small exploitation bias to help in flat regions? 
    # The -mu term already does this. 
    # Let's ensure finite values.
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
