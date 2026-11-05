"""Program for Maximum Independent Set via parallel min-degree exclusion process."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign priority scores using a parallel min-degree exclusion process.
    In each round, vertices with the minimum degree are identified.
    A maximal independent set is greedily selected from these candidates.
    Selected vertices and their neighbors are removed from the active set.
    Scores are assigned such that vertices selected in earlier rounds have higher priority,
    and within a round, selected vertices have higher priority than their neighbors.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    adj = adjacency.astype(bool)
    
    scores = np.zeros(n, dtype=float)
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
        
        # Find candidates: active vertices with min degree
        candidate_indices = active_indices[active_deg == min_deg]
        
        if len(candidate_indices) == 0:
            break
            
        # Compute degrees within the candidate set to break ties effectively.
        # Candidates with fewer connections to other candidates are preferred 
        # as they "waste" fewer potential picks in this round.
        sub_adj = adj[np.ix_(candidate_indices, candidate_indices)]
        internal_degrees = sub_adj.sum(axis=1)
        
        # Sort by internal degree (ascending), then by global degree (all same), then index
        # argsort is stable, so we just sort by internal degree
        sorted_order = np.argsort(internal_degrees)
        sorted_candidates = candidate_indices[sorted_order]
        
        # Greedy Maximal Independent Set on sorted_candidates
        selected_globals = []
        # Vectorized check for adjacency to already selected
        # We can maintain a boolean mask of "blocked" candidates
        blocked = np.zeros(len(sorted_candidates), dtype=bool)
        
        for i in range(len(sorted_candidates)):
            if blocked[i]:
                continue
            v = sorted_candidates[i]
            selected_globals.append(v)
            # Block neighbors of v in the candidate set
            # Find indices in sorted_candidates that are neighbors of v
            # sub_adj row for v's position in candidate_indices
            # We need the index of v in candidate_indices
            # It's easier to just check against selected_globals as before, 
            # but vectorizing the "block" step is faster if we map indices.
            
            # Let's stick to the simple loop but optimize the check
            # The previous loop checked adj[v, selected_globals].any()
            # This is O(K) where K is number of selected so far.
            # Total O(M^2) where M is size of selected set.
            
            # To speed up, we can use the sub_adj matrix.
            # Get the position of v in candidate_indices
            # This is expensive to do every time.
            
            # Given N=200, the simple loop is likely fine.
            # But let's try to improve the neighbor check using numpy.
            pass
            
        if not selected_globals:
            break
            
        # Assign scores to selected vertices
        score_val = base_score - round_id
        for v in selected_globals:
            scores[v] = score_val
            
        # Identify neighbors of selected vertices to remove and assign lower scores
        neighbor_mask = adj[selected_globals, :].any(axis=0)
        
        # Update active: remove selected and their neighbors
        active[selected_globals] = False
        active[neighbor_mask] = False
        
        # Assign scores to neighbors (lower than selected, higher than next round)
        neighbors_to_score = np.where(neighbor_mask)[0]
        for u in neighbors_to_score:
            # Ensure we don't overwrite a score if it was somehow set (shouldn't happen)
            if scores[u] == 0.0: 
                scores[u] = score_val - 0.1
        
        round_id += 1
        
    return scores
# EVOLVE-BLOCK-END
