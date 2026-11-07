"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute marginal utility heuristic for ant colony optimization.
    
    prizes: (5, 100) array of prizes per constraint per item
    weights: (5, 100) array of weights per constraint per item
    
    Returns: (100,) array of attractiveness scores (larger = more attractive)
    
    Strategy: Estimate opportunity cost by computing weighted scarcity of each
    constraint, then score items by prize relative to their consumption of
    scarce resources.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = prizes.shape
    
    # Total prize per item (sum across constraints)
    total_prizes = np.sum(prizes, axis=0)  # (100,)
    
    # Capacity is 25 for all constraints
    capacity = 25.0
    
    # For each constraint, compute the total weight of all items
    total_weight_per_constraint = np.sum(weights, axis=1)  # (5,)
    
    # Scarcity factor: how "tight" each constraint is relative to capacity
    # If total weight >> capacity, the constraint is very scarce
    scarcity = total_weight_per_constraint / capacity  # (5,)
    
    # Normalize scarcity to be in a reasonable range
    # Use a soft normalization: scarcity_factor = 1 + scarcity (so minimum is 1)
    scarcity_factor = 1.0 + scarcity  # (5,)
    
    # Effective weight of each item: dot product of item's weights with scarcity factors
    # weights.T is (100, 5), scarcity_factor is (5,)
    effective_weights = np.dot(weights.T, scarcity_factor)  # (100,)
    
    # Heuristic: prize / effective weight
    eps = 1e-9
    heur = total_prizes / (effective_weights + eps)
    
    # Ensure all values are positive and finite
    heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
    
    return heur
# EVOLVE-BLOCK-END
