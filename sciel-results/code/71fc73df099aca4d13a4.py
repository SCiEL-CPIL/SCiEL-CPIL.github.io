"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Returns an n x n matrix of attractiveness for placing item j right after item i.
    
    Strategy: Model heuristic as derived from FFD bin assignment structure.
    - Sort items descending
    - Assign to bins via First-Fit-Decreasing
    - For items in the same bin, high attractiveness (they form a valid cluster)
    - Attractiveness weighted by how well the pair fits (complementarity)
    - Pairs that don't fit get 1e-9
    """
    n = len(demands)
    demands = np.asarray(demands, dtype=np.float64)
    
    # Sort indices by demand descending for FFD
    order = np.argsort(-demands)
    sorted_demands = demands[order]
    
    # First-Fit-Decreasing bin assignment
    bins = []  # list of lists of original indices
    bin_residuals = []  # remaining capacity per bin
    
    for idx in order:
        placed = False
        for b in range(len(bins)):
            if bin_residuals[b] >= demands[idx]:
                bins[b].append(idx)
                bin_residuals[b] -= demands[idx]
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_residuals.append(capacity - demands[idx])
    
    # Build bin membership: bin_id for each item
    bin_of = np.full(n, -1, dtype=np.int64)
    for b, members in enumerate(bins):
        for idx in members:
            bin_of[idx] = b
    
    # Build heuristic matrix
    H = np.full((n, n), 1e-9, dtype=np.float64)
    
    # For each pair (i, j) where i != j:
    # - If they fit together (demands[i] + demands[j] <= capacity):
    #   - If they are in the same FFD bin: high attractiveness
    #   - Otherwise: moderate attractiveness based on complementarity
    # - If they don't fit: 1e-9
    
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    
    # Base complementarity score: prefer pairs that leave small residual
    # residual = capacity - (d_i + d_j), smaller is better (tighter pack)
    residual = capacity - pair_sum
    # Normalize: residual in [0, capacity - 40] roughly
    # Use inverse: lower residual -> higher score
    # Score = 1 / (1 + residual) gives tight pairs higher weight
    comp_score = 1.0 / (1.0 + residual)
    
    # Same-bin bonus: items in the same FFD bin get a multiplier
    same_bin = (bin_of[:, None] == bin_of[None, :])
    same_bin_bonus = np.where(same_bin, 2.0, 1.0)
    
    # Combine: only for fitting pairs
    H = np.where(fits, comp_score * same_bin_bonus, 1e-9)
    
    # Ensure diagonal is not used (i != j), set to 1e-9
    np.fill_diagonal(H, 1e-9)
    
    # Ensure all values are positive (clip very small values)
    H = np.maximum(H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
