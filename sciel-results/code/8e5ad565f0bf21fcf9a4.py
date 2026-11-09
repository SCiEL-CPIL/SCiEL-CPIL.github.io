"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Args:
        mu: Mean of GP predictions (shape 2000,)
        sigma: Std dev of GP predictions (shape 2000,)
        f_best: Best observed function value (scalar)
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        Scores (shape 2000,), higher is better.
    """
    # Explore-exploit balance parameter
    beta = 2.5
    
    # LCB for minimization: mu - beta * sigma
    # We want to MINIMIZE LCB, so we MAXIMIZE -LCB = beta * sigma - mu
    # Divide by cost to account for cost-awareness
    scores = (beta * sigma - mu) / cost
    
    # Ensure finite scores
    # If any are inf/nan, replace with 0 (worst case)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
