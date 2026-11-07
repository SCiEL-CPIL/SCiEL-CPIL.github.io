"""Cost-aware acquisition for 2D Ackley BO (Cost-Adjusted UCB)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 300.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score using Cost-Adjusted UCB.
    Higher is better (means "evaluate next").
    
    score = (-mu + beta * sigma) / cost
    
    We use a fixed beta tuned for the 30-cost budget on 2D Ackley.
    """
    # Deadline guard: If we are extremely close to the limit, 
    # we could potentially return a safe fallback, but usually 
    # the GP fitting is the bottleneck, not this vectorized op.
    # We just ensure this function is fast.
    
    n = mu.shape[0]
    eps = 1e-9

    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=eps, posinf=1e6, neginf=eps)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    cost = np.maximum(cost, eps)
    sigma = np.maximum(sigma, eps)

    # UCB Parameter
    # beta controls exploration. 
    # For Ackley 2D with limited budget, moderate exploration is key.
    # We use beta = 1.0 as a baseline structural change from EI.
    beta = 1.0
    
    # Calculate UCB score: Exploit (-mu) + Explore (beta * sigma)
    # Note: We want to minimize f, so lower mu is better. 
    # Acquisition should be higher for lower mu.
    # So we use -mu.
    ucb_score = -mu + beta * sigma
    
    # Cost-normalized UCB
    scores = ucb_score / cost

    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
# EVOLVE-BLOCK-END
