"""Improved heuristic for ant colony offline bin packing using matching-guided tightness."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing problem.
    
    Strategy:
    1. Compute a base tightness heuristic: for feasible pairs, reward near-full bins
       using a smooth function of remaining space (inverse-gap style).
    2. Use maximum weight matching (Hungarian algorithm) on a bipartite graph of
       items to identify a set of globally optimal disjoint pairs.
    3. Boost the heuristic for matched pairs to guide ACO toward proven-good pairings.
    4. Combine with a size-complementarity signal to handle items that don't pair well
       with each other but fit well with others.
    
    Args:
        demands: Array of item sizes.
        capacity: Bin capacity.
        
    Returns:
        Matrix of shape (N, N) where entry (i, j) is the attractiveness
        of placing item j after item i.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s kill
    
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    cap = float(capacity)
    
    if n == 0:
        return np.empty((0, 0))
    
    # Precompute pairwise sums
    pair_sums = d[:, None] + d[None, :]
    feasible = pair_sums <= cap
    
    # Remaining space after placing both items (for feasible pairs)
    remaining = np.where(feasible, cap - pair_sums, cap)
    
    # Base tightness: reward pairs that use more capacity
    # Use a smooth inverse-gap function with damping to avoid extreme values
    eps = 1e-6
    gap = remaining + eps
    
    # Primary signal: inverse of remaining space, scaled to be in a reasonable range
    # For feasible pairs, remaining is in [0, cap], so gap is in [eps, cap+eps]
    # 1/gap ranges from 1/(cap+eps) to 1/eps (very large)
    # Let's normalize: use 1/(1 + remaining/cap) which is in [0.5, 1] for feasible pairs
    # This is smoother and avoids extreme values
    
    # Actually, let's use a combination:
    # - Tightness: 1/(1 + remaining/cap) - rewards filling bins
    # - Size product: d_i * d_j / cap^2 - rewards large-large pairs (they're harder to place)
    # - Complementarity: 1/(1 + |d_i - d_j|/cap) - rewards similar sizes (less fragmentation)
    
    # Compute components
    tightness = 1.0 / (1.0 + remaining / cap)  # in [0.5, 1] for feasible
    size_product = (d[:, None] * d[None, :]) / (cap * cap)  # in [0, 1]
    size_diff = np.abs(d[:, None] - d[None, :])
    complementarity = 1.0 / (1.0 + size_diff / cap)  # in [0.5, 1]
    
    # Base heuristic for feasible pairs
    # Weighted combination emphasizing tightness
    base_heur = 0.5 * tightness + 0.3 * size_product + 0.2 * complementarity
    
    # Initialize full matrix
    heur = np.full((n, n), 1e-9)
    heur[feasible] = base_heur[feasible]
    
    # Now use matching to boost good pairs
    # Build a weight matrix for the Hungarian algorithm
    # We want to maximize total weight, so we use the negative of a cost
    # Cost = -tightness (we want to maximize tightness)
    # For infeasible pairs, set cost to a very large positive value (so they're avoided)
    
    # Check time budget before expensive matching
    if time.time() > deadline - 30:
        # Not enough time for matching, return base heuristic
        np.fill_diagonal(heur, 1e-9)
        heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
        return heur
    
    # Build cost matrix for Hungarian algorithm
    # We maximize tightness, so cost = -tightness for feasible pairs
    # For infeasible pairs, use a large cost
    cost_matrix = np.full((n, n), 1e6)
    cost_matrix[feasible] = -tightness[feasible]
    
    # Set diagonal to high cost (can't match item with itself in this context)
    np.fill_diagonal(cost_matrix, 1e6)
    
    # Solve maximum weight matching using Hungarian algorithm
    # linear_sum_assignment minimizes cost, so we pass negative weights
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Check time again
    if time.time() > deadline - 20:
        np.fill_diagonal(heur, 1e-9)
        heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
        return heur
    
    # Boost matched pairs
    # The matching gives us n pairs (i, col_ind[i])
    # For each matched pair, boost both heur[i, j] and heur[j, i]
    boost_factor = 2.0  # Strong boost for matched pairs
    
    for i, j in zip(row_ind, col_ind):
        if i != j and feasible[i, j]:
            heur[i, j] *= boost_factor
            heur[j, i] *= boost_factor
    
    # Ensure diagonal is set to a small value (can't place item after itself)
    np.fill_diagonal(heur, 1e-9)
    
    # Final safety checks
    heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
    
    if heur.shape != (n, n):
        raise ValueError("Wrong shape")
    
    return heur
# EVOLVE-BLOCK-END
