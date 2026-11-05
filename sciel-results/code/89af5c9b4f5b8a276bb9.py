"""Program for Maximum Independent Set via parallel discrete-time exclusion process."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign priority scores using a parallel discrete-time exclusion process.
    In each round, vertices with the minimum active degree become eligible.
    Scores are assigned based on the earliest round a vertex becomes eligible,
    capturing dynamic interference patterns.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    adj = adjacency.astype(bool)
    
    scores = np.full(n, -1.0, dtype=float)
    active = np.ones(n, dtype=bool)
    
    round_id = 0
    base_score = n + 1.0
    
    while active.any():
        # Calculate degrees of active vertices considering only active neighbors
        active_mask = active.astype(np.int32)
        deg = (adj * active_mask).sum(axis=1)
        
        # Only consider active vertices
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
            
        active_deg = deg[active_indices]
        min_deg = active_deg.min()
        
        # Find all vertices with minimum degree - these become eligible this round
        eligible_indices = active_indices[active_deg == min_deg]
        
        if len(eligible_indices) == 0:
            break
            
        # Assign eligibility-based scores: earlier rounds get higher scores
        # Only assign if not already scored (first time becoming eligible)
        unassigned_eligible = eligible_indices[scores[eligible_indices] < 0]
        for v in unassigned_eligible:
            scores[v] = base_score - round_id
        
        # Now greedily select an independent set from the eligible vertices
        # Sort by degree (ascending) to prefer less connected vertices
        eligible_degrees = deg[eligible_indices]
        sorted_order = np.argsort(eligible_degrees)
        sorted_eligible = eligible_indices[sorted_order]
        
        selected_globals = []
        for v in sorted_eligible:
            # Check if v is adjacent to any already selected vertex in this round
            if selected_globals:
                if adj[v, selected_globals].any():
                    continue
            selected_globals.append(v)
            
        if not selected_globals:
            # This shouldn't happen, but break to avoid infinite loop
            break
            
        # Remove selected vertices and their neighbors from active set
        neighbor_mask = adj[selected_globals, :].any(axis=0)
        active[selected_globals] = False
        active[neighbor_mask] = False
        
        round_id += 1
    
    # Any vertices that never became eligible get the lowest score
    unassigned = scores < 0
    if unassigned.any():
        scores[unassigned] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
