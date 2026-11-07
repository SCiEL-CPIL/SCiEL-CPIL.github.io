"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # UCB acquisition with cost penalty
    # Score = (f_best - (mu + beta * sigma)) / cost^gamma
    # We want higher score for better candidates.
    # If mu + beta*sigma < f_best, score is positive.
    
    beta = 1.5
    gamma = 1.0
    
    # Ensure sigma is not zero to avoid division issues if we were dividing by sigma,
    # but here we just add it.
    # Ensure cost is positive.
    
    ucb = mu + beta * sigma
    # Normalize by cost
    scores = (f_best - ucb) / (cost ** gamma)
    
    # Handle any numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
