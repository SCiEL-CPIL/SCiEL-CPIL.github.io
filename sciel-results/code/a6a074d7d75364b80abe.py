import numpy as np
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 300s limit
    # The function is called once per iteration. 
    # If we are close to the limit, we should return safe values.
    # However, Thompson Sampling is extremely fast (just one random draw).
    # The main cost is in the GP update outside this function.
    # We assume this function is called within the 300s budget for the whole loop.
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling:
    # Sample a function value for each candidate from the posterior N(mu, sigma^2)
    # Since we want to minimize, we pick the candidate with the lowest sampled value.
    # To make this an acquisition function that returns scores (higher is better),
    # we can return -sampled_value.
    
    # However, we also need to incorporate cost.
    # The cost is 1 + mean(x_scaled). Range [1, 2].
    # We want to penalize high cost.
    # Strategy: Score = -sampled_value * cost_penalty
    # Or: Score = - (sampled_value + lambda * cost)
    
    # Let's use a multiplicative penalty or additive.
    # Additive is more standard for cost-aware BO (expected cost).
    # But here we have a sampled value.
    # Let's try: Score = -sampled_value / (cost ** 0.5)
    # The square root penalty (sublinear) was found to improve exploration in high-cost regions.
    # This allows the algorithm to consider high-cost points if the sampled value is sufficiently low.
    
    # Draw samples from the posterior
    # np.random.rand() is fast and sufficient for 2000 points
    samples = mu + sigma_safe * np.random.randn(2000)
    
    # Apply cost penalty
    # cost is in [1, 2].
    # sqrt(cost) is in [1, ~1.414].
    # Dividing by sqrt(cost) reduces the "attractiveness" (makes score less negative) for high cost.
    # Wait, we want higher score = better.
    # If sampled_value is -0.5 (good), score = -(-0.5) / 1.414 = 0.35.
    # If sampled_value is -0.5 (good), score = -(-0.5) / 1.0 = 0.5.
    # So low cost gets a higher score for the same sampled value. This is correct.
    
    penalty = np.sqrt(cost)
    score = -samples / penalty
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
