"""Program for Maximum Independent Set via parallel min-degree exclusion process."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign priority scores using a parallel min-degree exclusion process.
    In each round, vertices with the minimum degree in the active subgraph are identified.
    A maximal independent set is greedily selected from these candidates, prioritizing
    vertices with lower degree within the candidate set to minimize interference.
    Selected vertices and their neighbors are removed from the active set.
    Scores are assigned such that vertices selected in earlier rounds have higher priority.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    adj = adjacency.astype(bool)
    
    scores = np.zeros(n, dtype=float)
    active = np.ones(n, dtype=bool)
    
    round_id = 0
    base_score = n + 10.0
    
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
            
        # Greedy Maximal Independent Set on candidate_indices
        # Heuristic: Pick candidate with lowest degree in the subgraph induced by remaining candidates
        selected_globals = []
        remaining_candidates = list(candidate_indices)
        
        while remaining_candidates:
            # Find the candidate with the minimum degree within the remaining candidates
            min_local_deg = float('inf')
            best_candidate = -1
            
            # Optimization: If the set is small, loop is fine.
            # For each v, count neighbors in remaining_candidates
            for v in remaining_candidates:
                local_deg = adj[v, remaining_candidates].sum()
                if local_deg < min_local_deg:
                    min_local_deg = local_deg
                    best_candidate = v
                    if min_local_deg == 0:
                        break
            
            if best_candidate == -1:
                break
                
            selected_globals.append(best_candidate)
            
            # Remove best_candidate and its neighbors from remaining_candidates
            # Create a boolean mask for remaining_candidates
            # adj[best_candidate, remaining_candidates] gives bool array of neighbors
            is_neighbor = adj[best_candidate, remaining_candidates]
            # We want to keep u if u is not best_candidate and not a neighbor
            # best_candidate is in remaining_candidates
            # Let's just filter
            new_remaining = []
            for idx, u in enumerate(remaining_candidates):
                if u == best_candidate:
                    continue
                if is_neighbor[idx]:
                    continue
                new_remaining.append(u)
            remaining_candidates = new_remaining
            
        if not selected_globals:
            break
            
        # Assign scores to selected vertices
        score_val = base_score - round_id
        
        for v in selected_globals:
            scores[v] = score_val
            
        # Identify neighbors of selected vertices to remove
        neighbor_mask = np.zeros(n, dtype=bool)
        if len(selected_globals) > 0:
            neighbor_mask = adj[selected_globals, :].any(axis=0)
        
        # Update active: remove selected and their neighbors
        active[selected_globals] = False
        active[neighbor_mask] = False
        
        # Assign scores to neighbors (lower than selected, higher than next round)
        neighbors_to_score = np.where(neighbor_mask)[0]
        for u in neighbors_to_score:
            if scores[u] == 0.0: 
                scores[u] = score_val - 0.1
        
        round_id += 1
        
    return scores
# EVOLVE-BLOCK-END
