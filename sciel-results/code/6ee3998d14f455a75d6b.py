import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Generate all weight-4 vectors: C(11,4)*16 = 330*16 = 5280
    w4_vectors = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            w4_vectors.append((combo, signs))
    
    n_w4 = len(w4_vectors)
    
    # Construct matrix for fast dot products
    w4_arr = np.zeros((n_w4, D), dtype=np.int8)
    for idx, (combo, signs) in enumerate(w4_vectors):
        for pos, (coord, sign) in enumerate(zip(combo, signs)):
            w4_arr[idx, coord] = sign
            
    # Dot products: n_w4 x n_w4
    # Conflict if dot >= 3 (since norm is 2, dist^2 = 8 - 2*dot >= 4 => dot <= 2)
    dot_products = w4_arr.astype(np.int16) @ w4_arr.astype(np.int16).T
    conflict = (dot_products >= 3).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Degrees
    degrees = conflict.sum(axis=1)
    
    best_set = []
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Greedy Min-Degree with random tie-breaking
    # We run multiple trials.
    # Optimization: Instead of rebuilding the graph every step, we maintain an active mask.
    # However, updating degrees dynamically in NumPy is tricky without loops.
    # Given N=5280, a simple loop over N steps, each doing a vectorized min on active nodes, is O(N^2) total.
    # 5280^2 ~ 28M ops. This is very fast in NumPy.
    
    for trial in range(200):
        if time.time() > deadline:
            break
            
        active = np.ones(n_w4, dtype=bool)
        selected = []
        
        # To introduce randomness in tie-breaking, we can add small random noise to degrees initially
        # or just pick randomly from min-degree nodes.
        
        while active.any():
            # Get degrees of active nodes
            active_degrees = np.where(active, degrees, np.inf)
            
            # Find min degree
            min_deg = np.min(active_degrees)
            
            # Find all nodes with this min degree
            candidates = np.where(active_degrees == min_deg)[0]
            
            if len(candidates) == 0:
                break
                
            # Pick one randomly
            idx = rng.choice(candidates)
            
            selected.append(idx)
            
            # Update active and degrees
            # Remove idx and its neighbors from active
            neighbors = conflict[idx].astype(bool)
            active[neighbors] = False
            active[idx] = False
            
            # Update degrees for remaining active nodes
            # degree of u decreases by 1 if u was connected to idx
            # We only care about active nodes
            # degrees[active] -= conflict[active, idx]? No, conflict is symmetric.
            # degrees[u] = sum(conflict[u, v] for v in active)
            # This update is O(N) per step. Total O(N^2).
            
            # Efficient update:
            # The degree of an active node u changes only if it was connected to the removed node idx.
            # If connected, its degree decreases by 1.
            # But wait, the "degree" in the greedy algorithm is the number of neighbors *in the current active set*.
            # When we remove idx, for every active neighbor u of idx, deg(u) decreases by 1.
            # Also, deg(idx) is removed.
            
            # We need to update `degrees` for the active nodes.
            # neighbors are those with conflict[idx, u] == 1.
            # For u in neighbors AND active: degrees[u] -= 1.
            
            # Vectorized update:
            # mask = active & neighbors
            # degrees[mask] -= 1
            
            mask = active & neighbors
            degrees[mask] -= 1
            
        current_size = len(selected)
        if current_size > best_size:
            best_size = current_size
            best_set = selected.copy()
            
    # Build result
    if not best_set:
        # Fallback
        result = np.zeros((1, D), dtype=np.int64)
        result[0, 0] = 1
        result[0, 1] = 1
        result[0, 2] = 1
        result[0, 3] = 1
        return result

    result = np.zeros((best_size, D), dtype=np.int64)
    for i, idx in enumerate(best_set):
        combo, signs = w4_vectors[idx]
        for pos, (coord, sign) in enumerate(zip(combo, signs)):
            result[i, coord] = sign
            
    return result
