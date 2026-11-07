"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: The evaluator kills after 330s. 
# We assume the acquisition function is called repeatedly within the loop.
# We need to ensure that if we are getting close to the time limit, 
# we don't hang. However, since we don't know the global start time 
# reliably from inside the function without state, we rely on the 
# fact that the operation itself must be fast.
# A simple Thompson sample is O(N) which is very fast.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Strategy:
    1. Sample a value from the GP posterior at each candidate point: 
       f_sampled = mu + sigma * z, where z ~ N(0, 1).
    2. To make it cost-aware, we can interpret the "effective value" 
       as the sampled value adjusted by the cost. 
       However, standard Thompson Sampling minimizes the sampled objective.
       To incorporate cost, we can use a cost-adjusted sample:
       score = - (f_sampled / cost)  ? No, that mixes units.
       
       A better approach for cost-aware Thompson Sampling in this context:
       We want to find a point that likely has a low true value f(x) 
       but also has a low cost.
       
       Let's stick to the standard TS principle but adjust for the 
       objective of minimizing f(x). 
       The acquisition value should be high if the sampled value is low 
       (good for minimization) and if the cost is low.
       
       Actually, the task says "Return 2000 finite scores (higher = evaluate next)".
       So we want to maximize the score.
       
       If we sample f_hat from N(mu, sigma^2), we prefer low f_hat.
       So -f_hat is a good score for minimization.
       
       How to include cost?
       Cost is 1 + mean(x_scaled).
       Dividing by cost is a common heuristic for cost-aware optimization 
       (Expected Improvement per unit cost).
       
       Score = - (f_sampled / cost)
       
       Let's verify the logic:
       - If f_sampled is very low (negative or close to 0), -f_sampled is high/positive.
       - If cost is high, the score is penalized.
       
       This seems reasonable.
    """
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check for sigma
    sigma = np.maximum(sigma, 1e-8)
    
    # Sample from the posterior: f_sampled ~ N(mu, sigma^2)
    # Use standard normal random numbers
    z = np.random.standard_normal(mu.shape)
    f_sampled = mu + sigma * z
    
    # Cost-aware score: We want to minimize f(x). 
    # So we maximize -f(x).
    # We also want to minimize cost, so we divide by cost.
    # Score = -f_sampled / cost
    
    # Handle potential division issues (cost should be >= 1.0)
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = -f_sampled / cost_safe
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
