"""Guided local search penalty rule for TSP with adaptive kick magnitude."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracking for safety
_START_TIME = time.monotonic()
_DEADLINE = 300.0  # 30 seconds buffer before the 330s kill limit

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Returns a modified distance matrix for the evaluator's 2-opt.
    
    Strategy:
    1. Estimate the current round number from the max value in edge_n_used.
    2. Calculate an adaptive penalty strength that starts high (to escape local optima)
       and decays as rounds progress (to allow refinement).
    3. Apply a uniform penalty to all edges in the current local_opt_tour.
    4. Apply a smaller, history-based penalty to edges that have been used frequently
       in previous tours (edge_n_used).
    5. Ensure the output is a valid, finite distance matrix.
    """
    # Safety check: if we are running out of time, return the original matrix to avoid timeout
    if time.monotonic() - _START_TIME > _DEADLINE:
        return np.asarray(edge_distance, dtype=float).copy()

    D = np.asarray(edge_distance, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        return D.copy()
        
    n = D.shape[0]
    
    # 1. Estimate round index
    # edge_n_used is accumulated over rounds. 
    # In round 1, max is likely 1. In round 2, max is likely 2, etc.
    # If the tour changes, counts might not increase uniformly, but max is a good proxy for "how many rounds have passed".
    max_used = np.max(edge_n_used)
    if max_used < 1:
        round_idx = 1
    else:
        round_idx = int(max_used)
    
    # 2. Adaptive Penalty Strength
    # We want a strong kick initially. 
    # Let's say the kick is proportional to the median edge length.
    # Median edge length is a robust scale factor.
    # We extract upper triangle to get unique edges for median calculation.
    triu_indices = np.triu_indices(n, k=1)
    edge_lengths = D[triu_indices]
    med_len = np.median(edge_lengths)
    
    # Base penalty: Start around 0.5 * med_len and decay.
    # Decay factor: 0.95^round_idx
    # This ensures we don't over-penalize in later rounds where the tour is likely good.
    decay = 0.95 ** round_idx
    base_penalty = 0.5 * med_len * decay
    
    # 3. Create the modified distance matrix
    D_mod = D.copy()
    
    # Identify edges in the current local_opt_tour
    # local_opt_tour is a list of nodes in order.
    # Edge (i, j) is in the tour if j is the next node after i.
    # Also (i, 0) is the last edge.
    
    # Initialize a penalty mask
    penalty_mask = np.zeros_like(D_mod)
    
    # Apply uniform penalty to all edges in the current tour
    # This forces the 2-opt to break the current structure.
    for i in range(n):
        j = local_opt_tour[(i + 1) % n]
        penalty_mask[i, j] = 1.0
        penalty_mask[j, i] = 1.0
    
    # Apply the base penalty to these edges
    D_mod[penalty_mask == 1.0] += base_penalty
    
    # 4. Additional history-based penalty
    # Edges that have been used many times in the past get an extra penalty.
    # This helps escape cycles of repeating the same bad edges.
    # Scale: 0.1 * med_len * (edge_n_used / max_used)
    # We only apply this to edges that are NOT in the current tour? 
    # Actually, applying it to all edges is safer. If an edge is in the current tour,
    # it gets base_penalty + history_penalty.
    
    # Normalize edge_n_used to [0, 1]
    if max_used > 0:
        norm_used = edge_n_used / max_used
    else:
        norm_used = np.zeros_like(edge_n_used)
        
    # History penalty factor
    history_factor = 0.2 * med_len * decay
    D_mod += history_factor * norm_used
    
    # 5. Ensure symmetry and validity
    D_mod = (D_mod + D_mod.T) / 2.0
    np.fill_diagonal(D_mod, 0.0)
    
    # Ensure finite values
    if not np.all(np.isfinite(D_mod)):
        return D.copy()
        
    return D_mod
# EVOLVE-BLOCK-END
