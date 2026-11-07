import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing algorithm.
    
    The heuristic eta[i, j] represents the attractiveness of placing item j
    immediately after item i in the same bin.
    
    Strategy:
    Composite heuristic that balances:
    1. Tight fit: minimize remaining gap after placing i+j
    2. Item size: prefer larger items to reduce stranded large items
    3. Bin utilization: reward pairs that fill a high fraction of capacity
    4. Complementarity signal: rank items by scarcity of valid partners
    
    Formula: eta[i, j] = item_size_j * exp(-alpha * gap) * (1 + beta * utilization) * scarcity_j
    where gap = capacity - w_i - w_j, and scarcity_j is the inverse degree in the compatibility graph.
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    
    # Calculate sums of all pairs
    pair_sums = W[:, None] + W[None, :]
    
    # Calculate gaps
    gaps = capacity - pair_sums
    
    # Identify valid pairs
    fits = gaps >= 0
    
    # Initialize heuristic matrix with a very small value for invalid pairs
    H = np.full((n, n), 1e-9)
    
    if np.any(fits):
        # Parameters tuned for balance between tight-fit and item-size preferences
        alpha = 0.05  # Controls how strongly we penalize large gaps
        beta = 2.0    # Controls how strongly we reward high utilization
        
        # Exponential decay of attractiveness with gap size
        # This creates a smooth preference for tight fits
        gap_penalty = np.exp(-alpha * gaps)
        
        # Utilization factor: how much of the bin is filled by i+j
        utilization = pair_sums / capacity
        
        # Item size factor: prefer placing larger items
        # W[None, :] broadcasts to (n, n) for column j
        item_size = W[None, :]
        
        # Complementarity signal: for each item j, count how many items i can fit with j
        # This is the degree in the compatibility graph (excluding self if not fitting)
        # fits[i, j] is True if item i and j can be in the same bin
        # Degree of j = number of i such that fits[i, j] is True
        degrees = np.sum(fits, axis=0)  # shape (n,)
        
        # Scarcity: inverse of degree, items with fewer compatible partners are more "scarce"
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        scarcity = 1.0 / (degrees + epsilon)
        
        # Normalize scarcity to be centered around 1 for stability
        scarcity_mean = np.mean(scarcity)
        scarcity_normalized = scarcity / scarcity_mean if scarcity_mean > 0 else np.ones(n)
        
        # Composite: reward tight fits, large items, and scarce items
        composite = item_size * gap_penalty * (1.0 + beta * utilization) * scarcity_normalized[None, :]
        
        # Set invalid pairs to 1e-9
        H = np.where(fits, composite, 1e-9)
        
    # Final safety checks: ensure finite and positive
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.where(H > 0, H, 1e-9)
    
    return H
