import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 300s limit, return safe values if exceeded
    # Although this is called once per evaluation step, the loop is controlled externally.
    # However, if the function itself is slow, we can't really "return best found" easily 
    # without state. But the task says "Each evaluation must finish within 300s".
    # The acquisition function is called once per iteration of the outer loop.
    # The outer loop runs while cost < 30.
    # Max iterations ~ 30.
    # 2000 points per iteration.
    # Computation is vectorized, should be very fast (<1ms).
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z-score for EI
    # For minimization, EI = E[max(f_best - f(x), 0)]
    # f(x) ~ N(mu, sigma^2)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-awareness:
    # Cost is 1 + mean(x scaled to [0,1]).
    # We want to favor low cost.
    # The current seed program uses EI / cost.
    # Let's try a different formulation:
    # Score = EI / (cost * cost) ? Or EI * (1/cost) - penalty?
    # Inspiration 3 mentioned "sublinear (square-root) cost penalty" or similar.
    # Let's try: Score = EI / (1 + 0.5 * cost) or EI * (2 - cost)?
    # Cost range is [1, 2].
    # If cost=1, weight=1. If cost=2, weight=0.5.
    # Let's use a linear cost penalty in the denominator: EI / (cost + 0.5)
    # This is softer than EI/cost.
    
    # Additionally, add an exploration term.
    # UCB style: EI + beta * sigma
    # But we need to balance cost.
    # Score = (EI + 0.1 * sigma_safe) / (cost + 0.5)
    
    # Let's refine the exploration coefficient.
    # In 8D, exploration is crucial.
    # Let's try:
    exploration_weight = 0.1
    cost_denom = cost + 0.5
    
    score = (ei + exploration_weight * sigma_safe) / cost_denom
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
