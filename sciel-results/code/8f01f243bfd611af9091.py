import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # UCB for minimization: minimize mu - beta * sigma
    # We want to maximize the "improvement" score.
    # Score proportional to (f_best - (mu - beta * sigma))
    # = f_best - mu + beta * sigma
    
    # Clip sigma to avoid numerical issues, though GP noise floor is 1e-6
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Beta value for exploration
    # A fixed beta of 2.0 or 3.0 is often good for 2D problems with small budgets.
    # Let's try 2.5.
    beta = 2.5
    
    # Calculate UCB improvement potential
    # If mu is very low (better than f_best), f_best - mu is positive.
    # The beta * sigma term adds exploration.
    improvement_potential = (f_best - mu) + beta * sigma_safe
    
    # Ensure non-negative?
    # If a point is predicted to be much worse than f_best, but has high uncertainty,
    # UCB might still rank it high. This is desired for exploration.
    # However, we should probably not evaluate points that are clearly bad and not uncertain.
    # But in BO, we evaluate the best candidate.
    
    # Normalize by cost
    scores = improvement_potential / cost
    
    # Handle NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # We want higher scores to be selected.
    # If improvement_potential is negative, it means mu > f_best + beta*sigma.
    # We should probably return 0 or a very small negative number for these?
    # The prompt says "Return 2000 finite scores (higher = evaluate next)".
    # It doesn't say they must be positive.
    # But if all scores are negative, the "best" is the least negative.
    
    return scores
