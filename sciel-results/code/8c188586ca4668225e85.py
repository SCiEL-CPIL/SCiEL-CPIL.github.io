"""Program for Maximum Independent Set via parallel discrete-time exclusion process."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign priority scores using a parallel discrete-time exclusion process.
    Vertices compete for selection; at each time step, the vertex with the fewest
    active neighbors is selected first. Selection timing captures dynamic interference:
    vertices selected early have high priority, and their neighbors are suppressed.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    adj = adjacency.astype(bool)
    
    scores = np.zeros(n, dtype=float)
    active = np.ones(n, dtype=bool)
    
    # Time step counter
    t = 0
    max_time = n  # Safety bound
    
    while active.any() and t < max_time:
        # Calculate active degrees for each vertex
        active_mask = active.astype(np.int32)
        active_deg = (adj * active_mask).sum(axis=1)
        
        # Only consider active vertices
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
        
        active_deg_active = active_deg[active_indices]
        
        # Find vertices with minimum active degree (fewest competing neighbors)
        min_deg = active_deg_active.min()
        candidates = active_indices[active_deg_active == min_deg]
        
        # Tie-breaking: prefer vertices with fewer total neighbors (globally "loose")
        total_deg = adj.sum(axis=1)
        candidate_total_deg = total_deg[candidates]
        sorted_order = np.argsort(candidate_total_deg)
        sorted_candidates = candidates[sorted_order]
        
        # Greedy selection from sorted candidates: pick non-adjacent vertices
        selected_in_round = []
        selected_set = set()
        for v in sorted_candidates:
            # Check if v is adjacent to any already selected in this round
            if selected_set:
                # Vectorized check: is v adjacent to any in selected_set?
                if adj[v, list(selected_set)].any():
                    continue
            selected_in_round.append(v)
            selected_set.add(v)
        
        if not selected_in_round:
            # Fallback: pick one candidate
            selected_in_round = [sorted_candidates[0]]
        
        # Assign scores based on time step (earlier = higher priority)
        # Use a decreasing base score with small perturbations within round
        base_score = float(n - t)
        for i, v in enumerate(selected_in_round):
            scores[v] = base_score - i * 0.01
        
        # Suppress neighbors of selected vertices
        neighbor_mask = np.zeros(n, dtype=bool)
        for v in selected_in_round:
            neighbor_mask |= adj[v]
        
        # Remove selected and their neighbors from active set
        active[selected_in_round] = False
        active[neighbor_mask & active] = False
        
        # Assign lower scores to suppressed neighbors (they lose the competition)
        suppressed = neighbor_mask & ~active
        for u in np.where(suppressed)[0]:
            if scores[u] == 0.0:
                scores[u] = base_score - len(selected_in_round) * 0.01 - 0.5
        
        t += 1
    
    return scores
# EVOLVE-BLOCK-END
