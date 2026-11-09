"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Strategy: Cost-adjusted Thompson Sampling.
    We sample from the posterior Gaussian N(mu, sigma^2) for each candidate.
    To account for cost, we penalize the sampled value by a term proportional 
    to the cost. This encourages the algorithm to pick candidates that are 
    both promising (low sampled value) and cheap (low cost).
    
    The penalty term is scaled by the current uncertainty (sigma) to ensure 
    that the exploration-exploitation trade-off adapts to the model's confidence.
    Specifically, we use: score = sampled_value - beta * cost / (sigma + eps)
    However, a more standard approach for cost-aware TS is to sample from the 
    posterior and then divide by cost, or subtract a cost penalty.
    
    Given the constraint that f_best is provided but not the full history, 
    and that we want to minimize the objective, we can use the sampled value 
    directly. But to be cost-aware, we should prefer cheaper evaluations.
    
    A robust cost-aware TS variant:
    1. Sample y ~ N(mu, sigma^2)
    2. Score = y / cost
    This effectively minimizes the expected value per unit cost.
    
    However, simply dividing by cost might be too aggressive if costs vary widely.
    Let's try: Score = y - lambda * cost.
    But we don't have a lambda.
    
    Alternative: The prompt says "cost-aware". The seed program used EI/cost.
    Let's stick to the principle: Higher score = more likely to be evaluated.
    If we sample y, lower y is better.
    So we want high scores for low y and low cost.
    
    Let's use: score = -y / cost.
    This is equivalent to minimizing y/cost.
    
    To avoid division by zero or near-zero, we add a small epsilon to cost.
    Cost is 1 + mean(x), so min cost is 1. No division by zero issues.
    
    We also need to ensure the scores are finite.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues in sampling
    sigma = np.clip(sigma, 1e-9, None)
    
    # Sample from the posterior
    # Use a fixed random seed? No, the function should be stochastic or 
    # use a global RNG. Since it's called in a loop, we should use np.random.
    # To make it deterministic per call if needed, we could use a hash, but 
    # standard TS uses random sampling.
    # The problem statement says "five seeded instances", implying the whole run is seeded.
    # We should use the global numpy random state.
    
    samples = mu + sigma * np.random.randn(mu.shape)
    
    # Cost-aware scoring:
    # We want to minimize the objective.
    # Higher score means "more attractive".
    # So score = - (sampled_value) / cost
    # This rewards low objective values and low costs.
    
    scores = -samples / cost
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
