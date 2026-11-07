"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes and weights are shape (5, 100)
    # We assume the total prize for an item is the sum of its components
    total_prizes = np.sum(prizes, axis=0)
    
    # Calculate the total weight per constraint to estimate scarcity/shadow price
    # Constraints with higher total weight are "tighter" or more congested, 
    # so resources in those dimensions are more valuable (higher shadow price).
    constraint_sums = np.sum(weights, axis=1) # Shape (5,)
    
    # Calculate the weighted cost for each item
    # cost_j = sum_k (weights[k, j] * constraint_sums[k])
    # This penalizes items that consume resources in constraints that are globally heavy.
    item_costs = weights.T @ constraint_sums # Shape (100,)
    
    # Heuristic: Prize / Cost
    # Add epsilon to avoid division by zero
    heur = total_prizes / (item_costs + 1e-9)
    
    return heur
# EVOLVE-BLOCK-END
