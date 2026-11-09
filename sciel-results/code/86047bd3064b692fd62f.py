"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    # demands is a 1D array of floats
    # Create a 2D array of sums for all pairs (i, j)
    # pair[i, j] = demands[i] + demands[j]
    pair = demands[:, None] + demands[None, :]
    
    # Calculate remaining capacity for each pair
    # rem[i, j] = capacity - (demands[i] + demands[j])
    rem = capacity - pair
    
    # We only care about pairs that fit (rem >= 0)
    # For pairs that don't fit, we set heuristic to 1e-9 (or 0, but task says non-positive becomes 1e-9)
    # The task says: "Heuristic entries that are not positive become 1e-9"
    # So we need to ensure the final matrix has no non-positive values, replacing them with 1e-9.
    
    # Heuristic idea: Favor tighter packs.
    # Inverse of remaining space + 1 to avoid division by zero and scale appropriately.
    # If rem is 0, heuristic is 1/1 = 1.
    # If rem is 100, heuristic is 1/101.
    # This strongly prefers filling the bin.
    
    # Create a mask for valid pairs (rem >= 0)
    valid = rem >= 0
    
    # Initialize heuristic matrix with 1e-9 (invalid/default)
    heur = np.full((n, n), 1e-9, dtype=np.float64)
    
    # Calculate heuristic for valid pairs
    # Combine tightness (Best Fit) with a bias towards larger items (Decreasing)
    # This mimics Best Fit Decreasing (BFD) behavior.
    # Heuristic = (1 / (rem + 1)) * (demands[j])
    # This favors pairs that are tight AND where the next item j is large.
    
    # Extract demands for j for the valid pairs
    # valid is a boolean mask of shape (n, n)
    # We need demands[j] for each (i, j) where valid[i, j] is True.
    # demands is 1D.
    # We can broadcast demands to (1, n) and multiply by the mask?
    # Or simply use numpy indexing.
    
    # Get indices of valid pairs
    # Or simpler: create a matrix of demands[j]
    # demands_j = demands[None, :] -> shape (1, n), broadcasts to (n, n)
    demands_j = demands[None, :]
    
    # Calculate the new heuristic
    # Base tightness
    tightness = 1.0 / (rem[valid] + 1.0)
    # Size bias
    # We need to align demands_j with the valid mask.
    # It's easier to compute for the whole matrix and then mask.
    
    # Recompute for whole matrix to be safe, then mask
    # rem is (n, n)
    # valid is (n, n)
    
    # Avoid division by zero or negative rem in the calculation before masking
    # We already have valid mask.
    
    # Let's compute the full heuristic matrix first
    # Heur_full = (demands[None, :] / (rem + 1.0))
    # But where rem < 0, this is invalid.
    
    # We can use np.where
    # denom = rem + 1.0
    # heur_full = np.where(valid, demands[None, :] / denom, 1e-9)
    
    # However, if rem is negative, denom is small or negative.
    # We only care about valid.
    
    # Let's just compute on the valid slice as before, but multiply by the corresponding demand.
    # We need to get the demands for the columns j where valid is True.
    # This is tricky with boolean indexing on 2D arrays if we want to keep shape.
    
    # Alternative:
    # heur[valid] = (1.0 / (rem[valid] + 1.0)) * demands[valid_indices_j]
    
    # Let's get the column indices for the valid mask
    rows, cols = np.where(valid)
    heur[rows, cols] = (1.0 / (rem[rows, cols] + 1.0)) * demands[cols]
    
    # Ensure no non-positive values remain (though 1e-9 is positive)
    # The task says "Heuristic entries that are not positive become 1e-9".
    # Our calculation yields positive values for valid pairs.
    # Invalid pairs are already 1e-9.
    
    return heur
# EVOLVE-BLOCK-END
