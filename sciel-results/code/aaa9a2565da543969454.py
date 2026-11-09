"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)"""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Thompson Sampling.
    
    We sample a value from the posterior distribution for each candidate
    and use the negative sampled value (normalized by cost) as the score.
    Higher scores correspond to candidates that are likely to have lower
    function values, adjusted for evaluation cost.
    """
    n = len(mu)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Cost safe
    cost_safe = np.maximum(cost, 1e-6)
    
    # Sample from the posterior
    # Use a fixed seed? No, the task implies stochasticity is allowed or 
    # the evaluator runs multiple times. 
    # However, "Each function runs on five seeded instances".
    # The acquisition function itself should probably be deterministic 
    # or use a global state? 
    # The task says "Return 2000 finite scores".
    # If I use np.random, it will be different every call.
    # This is standard for TS.
    
    # To ensure reproducibility within a single run if needed, 
    # one might worry about randomness. But BO is inherently stochastic 
    # if using TS.
    # The prompt says "five seeded instances f_k(x)". This refers to the 
    # objective function instances, not the acquisition randomness.
    # The acquisition randomness is part of the algorithm.
    
    # Generate standard normal samples
    samples = np.random.standard_normal(n)
    
    # Sampled objective values
    y_samples = mu + sigma_safe * samples
    
    # We want to minimize y. 
    # So we maximize -y.
    # Normalize by cost.
    score = -y_samples / cost_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
