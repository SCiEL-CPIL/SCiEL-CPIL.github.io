"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 330 - 5  # Safety margin: kill program 5s before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Instead of maximizing Expected Improvement (EI), we sample from the 
    posterior distribution of the objective function at each candidate point 
    and divide by the cost. This encourages exploration of high-uncertainty 
    regions (where sigma is high) relative to their cost, helping to escape 
    local minima in multimodal functions like Ackley and Rastrigin.
    """
    # Deadline guard: if we are running out of time, return a simple greedy score
    if time.time() > _DEADLINE:
        # Fallback: prioritize low mu and low cost to finish quickly
        return -mu / (cost + 1e-9)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Thompson Sampling:
    # We want to select the point that is likely to have the best (lowest) value
    # given the uncertainty, adjusted for the cost of evaluation.
    # We sample from N(mu, sigma^2) and divide by cost.
    # Since we want to MINIMIZE the function, we want the SAMPLE to be LOW.
    # However, the acquisition function must return scores where HIGHER is BETTER.
    # So we return - (sample / cost).
    
    # Draw samples from the posterior
    # Using a fixed seed for reproducibility within the same state? 
    # No, Thompson sampling should be stochastic to explore.
    # But wait, the evaluator calls this function once per iteration with 2000 candidates.
    # If we use random samples, the choice is stochastic.
    # To ensure we don't just pick noise, we can use a single sample per candidate.
    
    samples = np.random.normal(mu, sigma_safe)
    
    # The value we are optimizing for is f(x). We want low f(x).
    # Cost is C(x). We want to maximize (Improvement) / Cost.
    # In Thompson Sampling for minimization, we typically sample f(x) ~ N(mu, sigma)
    # and pick the one with the lowest sampled value, possibly penalized by cost.
    # A common cost-aware TS formulation: score = - (sampled_value) / cost
    # Or: score = - (sampled_value + beta * cost)
    
    # Let's use the ratio: -sampled_value / cost
    # This balances finding low values against the expense of evaluating them.
    # Points with high uncertainty (high sigma) will have a wider range of sampled values.
    # Sometimes they will sample very low values, encouraging exploration.
    
    scores = -samples / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
# EVOLVE-BLOCK-END
