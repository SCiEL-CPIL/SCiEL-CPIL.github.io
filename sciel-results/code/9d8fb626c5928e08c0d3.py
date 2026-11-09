"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # LCB strategy: score = mu - beta * sigma
    # Higher score means lower mu (better) and/or higher sigma (more exploration).
    # We adapt beta based on the current cost.
    # Total budget is 30.
    # As cost increases, we should reduce exploration (beta) to exploit.
    # At cost=0, beta is high. At cost=30, beta is low.
    
    # Normalize cost to [0, 1] where 1 is the limit
    c_norm = np.clip(cost / 30.0, 0.0, 1.0)
    
    # Beta schedule:
    # Start with high beta (e.g., 2.0) to explore broadly.
    # End with low beta (e.g., 0.5) to exploit near the best found.
    # Linear decay is a simple robust choice.
    beta_start = 2.5
    beta_end = 0.5
    beta = beta_start - (beta_start - beta_end) * c_norm
    
    # Calculate LCB scores
    # We want to return scores where higher is better.
    # LCB minimizes mu - beta * sigma.
    # So score = -(mu - beta * sigma) = -mu + beta * sigma.
    scores = -mu + beta * sigma
    
    # Ensure finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
