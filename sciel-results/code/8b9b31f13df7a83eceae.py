"""
Guided local search with reward-penalty hybrid strategy.
This approach actively rewards edges that have been under-used in previous tours,
aiming to guide the 2-opt search toward unexplored regions of the tour space
rather than just pushing away from the current local optimum.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.monotonic()
_DEADLINE = 300.0  # 30 seconds buffer before the 330s kill limit

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Returns a modified distance matrix for the evaluator's 2-opt.
    
    Strategy:
    1. Estimate the current round number from the max value in edge_n_used.
    2. Identify 'promising' edges: those with usage counts significantly lower 
       than the expected count (round_idx). These are structurally under-explored.
    3. Apply a negative offset (reward) to these under-used edges to make them 
       more attractive to 2-opt.
    4. Apply a positive offset (penalty) to edges in the current local_opt_tour 
       to force diversification.
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
    # max_used is a proxy for how many rounds have passed.
    if edge_n_used.size == 0:
        return D.copy()
        
    max_used = np.max(edge_n_used)
    if max_used < 1:
        round_idx = 1
    else:
        round_idx = int(max_used)
    
    # 2. Calculate scale factor
    # Use median edge length as a robust scale.
    triu_indices = np.triu_indices(n, k=1)
    edge_lengths = D[triu_indices]
    med_len = np.median(edge_lengths)
    
    # 3. Identify under-used edges (candidates for reward)
    # An edge is "under-used" if its count is less than the current round index.
    # We want to reward edges that haven't been part of many tours.
    # Let's define a threshold: if count < round_idx * 0.5, it's significantly under-used.
    # However, in early rounds, most edges are under-used.
    # Let's be more selective: reward edges with count == 0 or very low.
    # Actually, the prompt suggests: "edges with usage counts below the current round index receive a small negative distance offset".
    # Let's interpret "below current round index" as count < round_idx.
    # But in round 1, round_idx=1, so count < 1 means count == 0.
    # In round 2, round_idx=2, so count < 2 means count 0 or 1.
    # This effectively rewards edges that are NOT in the current tour (since current tour edges get incremented).
    # Wait, edge_n_used is accumulated OVER the rounds.
    # The evaluator adds the current tour's edges to edge_n_used BEFORE calling this function?
    # "add the current tour's edges to edge_n_used, call your function... with ... edge_n_used"
    # So the current tour's edges have ALREADY been incremented.
    # Therefore, edges in the current tour have count >= 1.
    # Edges NOT in the current tour might have count < round_idx if they were unused in previous rounds.
    
    # Let's define the reward mask: edges where edge_n_used < round_idx.
    # Note: edge_n_used is symmetric.
    reward_mask = (edge_n_used < round_idx).astype(float)
    # Exclude diagonal (self-loops)
    np.fill_diagonal(reward_mask, 0.0)
    
    # 4. Identify current tour edges (candidates for penalty)
    penalty_mask = np.zeros_like(D)
    for i in range(n):
        j = local_opt_tour[(i + 1) % n]
        penalty_mask[i, j] = 1.0
        penalty_mask[j, i] = 1.0
        
    # 5. Calculate magnitudes
    # Reward magnitude: small negative offset.
    # We want to encourage 2-opt to pick these edges.
    # Let's use a fraction of the median length.
    reward_mag = 0.1 * med_len
    
    # Penalty magnitude: strong positive offset to break the current tour.
    # Let's use a larger fraction.
    penalty_mag = 0.5 * med_len
    
    # 6. Apply modifications
    D_mod = D.copy()
    
    # Apply reward to under-used edges
    # Only apply reward to edges that are NOT in the current tour?
    # If an edge is in the current tour, it has just been incremented, so its count is likely >= 1.
    # If round_idx > 1, it might still be < round_idx.
    # But we definitely want to penalize current tour edges to force change.
    # So, for edges in the current tour, we apply penalty.
    # For other edges, if they are under-used, we apply reward.
    
    # Create a mask for "under-used AND not in current tour"
    reward_apply_mask = reward_mask * (1.0 - penalty_mask)
    
    D_mod -= reward_mag * reward_apply_mask
    
    # Apply penalty to current tour edges
    D_mod += penalty_mag * penalty_mask
    
    # 7. Ensure symmetry and validity
    D_mod = (D_mod + D_mod.T) / 2.0
    np.fill_diagonal(D_mod, 0.0)
    
    # Ensure finite values
    if not np.all(np.isfinite(D_mod)):
        return D.copy()
        
    return D_mod
# EVOLVE-BLOCK-END
