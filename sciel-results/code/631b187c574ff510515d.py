"""Program for Maximum Independent Set via parallel min-degree exclusion process."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign priority scores using a parallel min-degree exclusion process.
    Vertices with the fewest active neighbors are selected in each round.
    Tie-breaking prefers vertices whose neighbors have higher degrees, 
    effectively removing 'dense' parts of the graph first to preserve 'sparse' parts.
    Scores are assigned based on the round index (earlier rounds get higher scores).
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    adj = adjacency.astype(bool)
    
    scores = np.zeros(n, dtype=float)
    active = np.ones(n, dtype=bool)
    
    round_id = 0
    # Use a large base to ensure round 0 > round 1 > ...
    # and enough separation to handle any number of rounds (max n)
    base_score = 100000.0
    step = 1.0
    
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
        
        # Find candidates: active vertices with min degree
        candidate_indices = active_indices[active_deg == min_deg]
        
        if len(candidate_indices) == 0:
            break
            
        # Tie-breaking: Calculate sum of degrees of active neighbors for each candidate
        # We prefer candidates with HIGHER neighbor degree sum (sacrifice dense areas)
        neighbor_degree_sums = np.zeros(len(candidate_indices))
        for i, v in enumerate(candidate_indices):
            # Get active neighbors of v
            neighbors = np.where(adj[v] & active)[0]
            if len(neighbors) > 0:
                neighbor_degree_sums[i] = deg[neighbors].sum()
            else:
                neighbor_degree_sums[i] = 0.0
        
        # Sort candidates by neighbor_degree_sum descending
        # If tie, maybe by index for stability?
        sorted_order = np.argsort(-neighbor_degree_sums)
        sorted_candidates = candidate_indices[sorted_order]
        
        # Greedy Maximal Independent Set on sorted_candidates
        selected_globals = []
        for v in sorted_candidates:
            # Check if v is adjacent to any already selected vertex in this round
            if selected_globals:
                if adj[v, selected_globals].any():
                    continue
            selected_globals.append(v)
            
        if not selected_globals:
            break
            
        # Assign scores to selected vertices
        score_val = base_score - round_id * step
        for v in selected_globals:
            scores[v] = score_val
            
        # Identify neighbors of selected vertices to remove
        neighbor_mask = adj[selected_globals, :].any(axis=0)
        
        # Update active: remove selected and their neighbors
        active[selected_globals] = False
        active[neighbor_mask] = False
        
        # Assign scores to neighbors (lower than selected, higher than next round)
        # We give them a score between current round and next round
        # score_val - step/2
        neighbors_to_score = np.where(neighbor_mask)[0]
        for u in neighbors_to_score:
            if scores[u] == 0.0: 
                scores[u] = score_val - step * 0.5
        
        round_id += 1
        
    return scores
# EVOLVE-BLOCK-END
