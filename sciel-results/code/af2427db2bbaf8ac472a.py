"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard variables
# The function is called many times, but the constraint is on the *evaluation* step.
# The prompt says "Each evaluation must finish within 300 s".
# The acquisition function itself is called once per step.
# We should ensure it's fast.
# Thompson Sampling is computationally very cheap: O(N) for sampling and argmin.

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling Acquisition Function.
    
    For each candidate point, we draw a single sample from the posterior
    predictive distribution: y ~ N(mu, sigma^2).
    We then return these samples as scores. Since the goal is minimization,
    the evaluator will pick the candidate with the lowest score (which corresponds
    to the most promising low function value in the sampled function).
    
    Wait, the contract says: "Return 2000 finite scores, higher meaning evaluate next".
    And "evaluates the candidate with the highest score".
    
    So if we want to minimize the function, we should return scores that are HIGH
    for points that are likely to have LOW function values.
    
    In Thompson Sampling for minimization:
    We sample f(x) ~ N(mu(x), sigma(x)^2).
    We want to find x that minimizes f(x).
    So we should select the x with the minimum sampled value.
    
    But the interface requires returning scores where HIGHER is better.
    So we should return -sampled_value.
    
    Let's verify:
    If mu = [-1, -2, -3] and sigma = [0.1, 0.1, 0.1]
    Sample: [-1.05, -1.95, -2.95]
    We want to pick the 3rd one (value -2.95) because it's the lowest (best for minimization).
    If we return -sampled_value: [1.05, 1.95, 2.95].
    The highest score is 2.95, which corresponds to the 3rd candidate.
    This is correct.
    
    Implementation:
    1. Draw random noise: z ~ N(0, 1) for each candidate.
    2. Compute sampled values: y = mu + sigma * z
    3. Return scores: -y
    
    This is a single-sample Thompson Sampling. It is stochastic, which is fine
    for BO and often leads to better diversity than deterministic EI.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-10)
    
    # Draw random samples from standard normal distribution
    # Using a fixed seed would make it deterministic, but TS is inherently stochastic.
    # The evaluator runs the whole loop for 3 seeds. The stochasticity inside the loop
    # is part of the method.
    # However, to ensure reproducibility if needed, or just to avoid potential issues
    # with random state, we can use np.random.
    # The task does not specify a random seed for the acquisition function itself,
    # so we use the global numpy random state.
    
    z = np.random.standard_normal(len(mu))
    
    # Sampled function values
    y = mu + sigma * z
    
    # Scores: higher is better for minimization, so we negate the sampled values
    scores = -y
    
    # Safety check: ensure all scores are finite
    # If mu or sigma contain NaN/Inf, this will catch it.
    scores[~np.isfinite(scores)] = -np.inf # Or 0.0? 
    # The contract says "Return 2000 finite scores".
    # So we must replace non-finite values with finite ones.
    # If a value is -inf (from -inf sample), it would be the "best" (highest) if we used -inf?
    # No, -(-inf) = inf. 
    # Let's just clamp to a large negative number or 0.
    # Actually, if mu or sigma are bad, the GP is bad.
    # Let's replace non-finite with 0.0 to be safe and neutral.
    scores[~np.isfinite(scores)] = 0.0
    
    return scores

# EVOLVE-BLOCK-END
