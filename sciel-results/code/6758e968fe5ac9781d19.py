"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Calculate priority scores for a greedy independent set decoder.
    
    Strategy:
    1. Use a multi-step greedy simulation to estimate how many vertices each
       candidate can contribute when selected in a greedy pass.
    2. Score = estimated contribution from selecting this vertex first,
       plus a bonus for low degree (fewer neighbors to block).
    
    The simulation: for each vertex v, simulate picking v first, then greedily
    pick the remaining vertices in order of (low degree, high neighbor degree).
    Count how many total vertices are picked. This gives an empirical estimate
    of the set size if v is prioritized first.
    
    For efficiency, we do a limited number of simulation steps per vertex.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Leave 10s margin
    
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    
    # Precompute degrees
    degrees = adj.sum(axis=1).astype(np.int32)
    
    # Precompute neighbor degree sums (sum of degrees of neighbors)
    neighbor_deg_sums = (adj @ degrees).astype(np.int32)
    
    # Initial ordering: sort by (degree ascending, neighbor_deg_sum descending)
    # Lower degree = easier to include; higher neighbor_deg_sum = blocking
    # high-degree vertices is beneficial
    order = np.lexsort((neighbor_deg_sum * -1, degrees))
    
    # Simulation: for each starting vertex, estimate the greedy set size
    scores = np.zeros(n, dtype=np.float64)
    
    # We'll simulate picking each vertex first, then continue greedily
    # For speed, limit the simulation length
    MAX_SIM_STEPS = min(n, 200)
    
    for start_v in range(n):
        if time.time() > DEADLINE:
            # Fallback: use a simpler heuristic
            break
        
        # Start with vertex start_v picked
        picked = {start_v}
        blocked = set(np.where(adj[start_v] == 1)[0].tolist())
        count = 1
        
        # Continue greedy: pick next vertex in order that's not blocked/picked
        for v in order:
            if time.time() > DEADLINE:
                break
            if v in picked or v in blocked:
                continue
            # Pick v
            picked.add(v)
            count += 1
            # Block its neighbors
            for u in np.where(adj[v] == 1)[0]:
                blocked.add(int(u))
            if count + len(order) - len(picked) - len(blocked) <= 0:
                break
        
        scores[start_v] = count
    
    # If we ran out of time, use a simpler heuristic for remaining vertices
    if time.time() > DEADLINE:
        # Simple heuristic: low degree, high neighbor degree sum
        inv_deg = 1.0 / (degrees.astype(np.float64) + 1.0)
        scores[~np.isfinite(scores)] = inv_deg[~np.isfinite(scores)]
        # For vertices that were scored, add a small degree bonus
        scored_mask = np.isfinite(scores) & (scores > 0)
        scores[scored_mask] += 0.01 * inv_deg[scored_mask]
    
    # Final priority: higher score = higher priority
    # We want the decoder to pick high-score vertices first
    return scores.astype(np.float64)
# EVOLVE-BLOCK-END
