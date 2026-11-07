"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes and weights are expected to be shape (5, 100)
    # Ensure they are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Capacity is fixed at 25 for all constraints
    capacity = 25.0
    
    # Calculate tightness for each constraint
    # Tightness = Total Weight in constraint / Capacity
    # Higher tightness means the constraint is more saturated
    sum_weights_per_constraint = weights.sum(axis=1)
    tightness = sum_weights_per_constraint / capacity
    
    # Calculate value density for each item in each constraint
    # density[c, i] = prizes[c, i] / weights[c, i]
    # Use epsilon to avoid division by zero
    epsilon = 1e-9
    densities = prizes / (weights + epsilon)
    
    # Weighted sum of densities
    # Items that have high value density in tight constraints get higher scores
    # tightness is shape (5,), densities is shape (5, 100)
    # We broadcast tightness to multiply each row of densities
    weighted_densities = tightness[:, np.newaxis] * densities
    
    # Sum across constraints to get a single score per item
    scores = weighted_densities.sum(axis=0)
    
    # Ensure the output is a 1D array of floats
    return scores
# EVOLVE-BLOCK-END
