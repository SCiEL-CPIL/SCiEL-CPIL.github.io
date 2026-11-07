"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB for minimization.
    # We maximize the potential improvement over the cost.
    # The optimistic estimate of the objective is mu - beta * sigma.
    # The potential improvement is f_best - (mu - beta * sigma).
    
    # Choose beta. 2.0 is a standard choice for exploration/exploitation balance.
    # In 2D Ackley, exploration is important.
    beta = 2.0
    
    # Calculate optimistic value
    # mu and sigma are arrays of shape (2000,)
    # Ensure sigma is non-negative (it should be, but just in case)
    sigma = np.maximum(sigma, 0.0)
    
    optimistic_value = mu - beta * sigma
    
    # Potential improvement
    # If optimistic_value < f_best, improvement is positive.
    potential_improvement = f_best - optimistic_value
    
    # Cost-aware score
    # cost is provided and > 0
    scores = potential_improvement / cost
    
    # Return finite scores
    # Ensure no NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
