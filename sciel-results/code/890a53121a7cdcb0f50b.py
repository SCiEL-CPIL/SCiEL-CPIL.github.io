"""Guided local search penalty for TSP; HARD CUT approach: force breaking of longest edges."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step: 
    Instead of soft penalties based on regret, we hard-disrupt the current tour by 
    significantly increasing the cost of the longest edges in the current tour. 
    This forces the subsequent 2-opt (run on our matrix) to break these edges, 
    effectively performing a large-scale move that escapes the local basin.
    """
    D = np.asarray(edge_distance, dtype=np.float64)
    tour0 = np.asarray(local_opt_tour, dtype=int)
    used = np.asarray(edge_n_used, dtype=np.float64)
    n = D.shape[0]

    t_start = time.time()
    T_LIMIT = 280.0 # Safety margin

    if n < 4:
        return D.copy()

    # 1. Identify edges in the current tour
    # Edge i is (tour[i], tour[i+1])
    edges = []
    lengths = []
    for i in range(n):
        u = tour0[i]
        v = tour0[(i + 1) % n]
        length = D[u, v]
        edges.append((u, v))
        lengths.append(length)
    
    lengths = np.array(lengths)
    
    # 2. Strategy: Penalize the top X% longest edges in the current tour.
    # We choose a dynamic percentage based on the number of times this tour (or similar) 
    # might have been seen, but since we only have edge counts, we use a fixed aggressive 
    # cut for the longest 20% of edges in the current tour.
    
    # To avoid getting stuck in a cycle where we always break the same long edges, 
    # we can randomize the selection among the top candidates if the usage count is high.
    # However, for a structural change, let's stick to a deterministic "Longest Edge" cut 
    # but combined with a "Usage-Aware" penalty for the rest.
    
    M = D.copy()
    
    # Get indices of the top 20% longest edges
    k = max(1, int(0.2 * n))
    # Sort edges by length descending
    sorted_indices = np.argsort(lengths)[::-1]
    
    # Penalty magnitude: 
    # We want the 2-opt to *definitely* break these edges.
    # A 2-opt move replacing edge (u,v) with (u,x) and (v,w) has delta 
    # D[u,x] + D[v,w] - D[u,v] - D[w,x].
    # If we add P to D[u,v], the delta becomes smaller by P.
    # To ensure breaking, P should be larger than the potential gain from keeping it.
    # A safe P is something like 2 * mean_distance.
    
    mean_dist = np.mean(D[D > 0]) if np.any(D > 0) else 1.0
    penalty_val = 2.0 * mean_dist
    
    # Apply hard penalty to top k edges
    for idx in sorted_indices[:k]:
        u, v = edges[idx]
        M[u, v] += penalty_val
        M[v, u] += penalty_val
        
    # 3. Apply soft penalty to other tour edges based on usage count
    # This encourages diversity for edges that aren't the "worst" but are "tired"
    max_used = np.max(used) if np.max(used) > 1e-10 else 1.0
    for i in range(n):
        if i in sorted_indices[:k]:
            continue
        u, v = edges[i]
        # Scale penalty by usage: more used = higher penalty
        # Cap the penalty to avoid distorting the matrix too much
        usage_pen = 0.5 * (used[u, v] / max_used) * mean_dist
        M[u, v] += usage_pen
        M[v, u] += usage_pen

    # Ensure symmetry and finiteness
    M = (M + M.T) * 0.5
    np.fill_diagonal(M, 0.0)
    
    # Check time
    if time.time() - t_start > T_LIMIT:
        # Return original if we somehow took too long (unlikely for this logic)
        return D

    M = np.where(np.isfinite(M), M, D)
    
    return M
# EVOLVE-BLOCK-END
