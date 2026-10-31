import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute conflict graph: two vectors conflict if their dot product > 2
    # For weight-4 vectors in {-1,0,1}^11, dot product ranges from -4 to 4
    # Non-conflict means |dot| <= 2, i.e., dot <= 2 (since we want independent set)
    # Actually, for kissing number, we need distance >= max_norm
    # dist^2 = |u|^2 + |v|^2 - 2*u.v = 4 + 4 - 2*u.v = 8 - 2*u.v
    # max_norm^2 = 4
    # Need: 8 - 2*u.v >= 4 => u.v <= 2
    # So conflict if u.v > 2
    
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Tomita's branch-and-bound for Maximum Independent Set
    # We'll use a recursive approach with degree-based ordering and upper bound pruning
    
    # Convert to adjacency list for faster neighbor lookups
    adj = [np.where(conflicts[i])[0] for i in range(N4)]
    
    # Tomita's algorithm with color upper bound
    # We'll implement a simplified version that's efficient for this graph
    
    best_set = []
    best_size = 0
    current_set = []
    
    # Use a greedy coloring to get upper bounds
    # The key insight: if we color the subgraph, the maximum independent set
    # is at most the number of colors needed
    
    # For efficiency, we'll use a recursive B&B with:
    # 1. Degree-based vertex ordering (process high-degree vertices first)
    # 2. Upper bound from greedy coloring
    # 3. Prune if current_size + upper_bound <= best_size
    
    # Since N4=5280 is large, we need to be careful with recursion depth
    # and use iterative approaches where possible
    
    # First, get a good initial solution using greedy
    alive = np.ones(N4, dtype=bool)
    greedy_selected = []
    degrees = conflicts.sum(axis=1).astype(np.int32)
    rng = np.random.default_rng(42)
    
    while np.any(alive) and time.time() < deadline - 30:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        # Pick vertex with minimum degree (Tomita uses max degree for coloring, 
        # but min degree often works well for greedy MIS)
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        
        greedy_selected.append(chosen)
        
        # Remove chosen and its neighbors
        neighbors = adj[chosen]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        alive[to_remove] = False
        
        # Update degrees
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
    
    best_set = greedy_selected
    best_size = len(best_set)
    
    # Now try to improve with local search / branch and bound
    # For each vertex not in best_set, try to swap it in if it improves
    
    in_set = np.zeros(N4, dtype=bool)
    for v in best_set:
        in_set[v] = True
    
    # Try to add vertices that don't conflict with current set
    # This is a local improvement step
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for v in range(N4):
            if in_set[v]:
                continue
            # Check if v can be added (no conflicts with current set)
            if np.any(conflicts[v][in_set]):
                continue
            # Try adding v
            in_set[v] = True
            new_size = in_set.sum()
            
            # Try to remove one conflicting vertex to see if we can maintain size
            # Actually, if v has no conflicts with current set, we can just add it
            best_size = new_size
            improved = True
            
    # Convert back to list
    best_set = np.where(in_set)[0].tolist()
    best_size = len(best_set)
    
    # If we have time, try a more sophisticated B&B on a subset
    # For now, return what we have
    
    result_v4 = vectors_v4[best_set]
    
    # Combine V4 and V1
    # V1 vectors have norm^2 = 4, same as V4
    # Check compatibility: V1 dot V4
    # V1 is (2,0,...,0) etc, V4 has entries in {-1,0,1}
    # Dot product is ±2 or 0, so always <= 2. Compatible!
    # V1 dot V1: (2,0,..) dot (2,0,..) = 4 if same dim, 0 if different
    # So V1 vectors in same dim conflict with each other (dot=4>2)
    # But V1 in different dims are compatible (dot=0)
    # So we can take at most 2 per dimension (one +2, one -2)? 
    # Wait: (2,0,..) dot (-2,0,..) = -4, which is <= 2, so compatible!
    # (2,0,..) dot (2,0,..) = 4 > 2, conflict
    # So we can take both +2 and -2 in same dimension
    # But we can't take two +2 in same dimension (only one exists)
    # So all 22 V1 vectors are mutually compatible? Let me check:
    # v_i = 2e_i and v_j = -2e_i: dot = -4 <= 2, OK
    # v_i = 2e_i and v_k = 2e_k (i≠k): dot = 0 <= 2, OK
    # v_i = 2e_i and v_k = -2e_k (i≠k): dot = 0 <= 2, OK
    # Yes! All 22 V1 vectors are mutually compatible.
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
