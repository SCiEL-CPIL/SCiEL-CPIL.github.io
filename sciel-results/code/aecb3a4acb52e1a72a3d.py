import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Structural Change: Switch from Expected Improvement (EI) to Upper Confidence Bound (UCB)
    # UCB balances exploration (sigma) and exploitation (mu) explicitly.
    # For minimization, we maximize -mu + beta * sigma.
    
    # Exploration parameter
    beta = 2.0
    
    # Base UCB score for minimization
    # We want to find low values, so -mu is high when mu is low (good).
    # Adding beta * sigma encourages exploring high uncertainty regions.
    ucb_score = -mu + beta * sigma
    
    # Additive Cost Penalty
    # The insight suggests additive penalties allow high-improvement candidates to overcome cost.
    # Cost is in [1, 2]. cost - 1 is in [0, 1].
    # We subtract a penalty proportional to the extra cost.
    # Scale 0.5 is a heuristic start for the trade-off.
    cost_penalty = 0.5 * (cost - 1.0)
    
    scores = ucb_score - cost_penalty
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
