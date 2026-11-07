# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time guard state
_start_time = time.time()
_TIME_LIMIT = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _TIME_LIMIT
    
    # Check elapsed time
    if time.time() - _start_time > _TIME_LIMIT:
        # Return a safe, finite score if we are running out of time
        # This should rarely trigger as the function is fast, but satisfies the directive
        return np.zeros_like(mu)

    # Cost-aware Lower Confidence Bound (LCB) for minimization
    # LCB = mu - beta * sigma
    # We divide by cost to get "improvement per unit cost" (lower is better for value, so we want high score for low value/cost)
    # Note: Standard LCB minimizes mu - beta*sigma. 
    # For an acquisition function where higher is better:
    # We want to select points with low mu and high sigma.
    # Score = (mu - beta*sigma) / cost is not quite right because we want to MAXIMIZE the score.
    # If we use LCB, we are looking for low values.
    # Let's use: Score = (f_best - (mu - beta*sigma)) / cost ? No, that mixes EI and LCB.
    
    # Pure LCB approach for maximization of score:
    # We want to find x that minimizes f(x).
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to pick x with the lowest LCB.
    # But the acquisition function must return scores where HIGHER is better.
    # So we can return: Score = -LCB(x) / cost(x) ? 
    # Or simply: Score = (f_best - LCB(x)) / cost(x) ? This is "Expected Improvement-like" but using LCB as the proxy for next value.
    
    # Let's stick to the standard "Cost-Effective LCB" which is often defined as:
    # CE-LCB = (f_best - (mu - beta*sigma)) / cost
    # This is structurally similar to CEI but uses the lower bound of the distribution instead of the expected value.
    # It tends to be more exploratory than EI for the same beta.
    
    beta = 2.0
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate LCB
    lcb = mu - beta * sigma
    
    # Calculate "Improvement" over f_best using LCB
    # This measures how much better the worst-case (beta-sigma) prediction is than the current best.
    improvement = f_best - lcb
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    scores = improvement / cost
    
    # Handle numerical instabilities
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
