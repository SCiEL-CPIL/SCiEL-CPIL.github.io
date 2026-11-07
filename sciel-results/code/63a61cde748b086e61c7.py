"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes and weights are expected to be shape (5, 100)
    # Ensure they are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # 1. Calculate total prize for each item
    total_prizes = prizes.sum(axis=0)
    
    # 2. Calculate the tightness of each constraint
    # Capacity is fixed at 25 for all constraints
    capacity = 25.0
    # Sum of weights for each constraint across all items
    sum_weights_per_constraint = weights.sum(axis=1)
    
    # Tightness factor: higher means the constraint is more saturated
    # The "cost" of using weight in a tight constraint is higher.
    tightness = sum_weights_per_constraint / capacity
    
    # 3. Calculate the "effective weight" of each item
    # An item's weight in constraint c is scaled by the tightness of that constraint.
    # This penalizes items that consume resources in already tight constraints.
    # effective_weight[i] = sum_c( weights[c, i] * tightness[c] )
    effective_weights = np.sum(weights * tightness[:, np.newaxis], axis=0)
    
    # 4. Calculate heuristic score
    # Score is total prize divided by effective weight
    epsilon = 1e-9
    scores = total_prizes / (effective_weights + epsilon)
    
    # Ensure the output is a 1D array of floats
    return scores
# EVOLVE-BLOCK-END
