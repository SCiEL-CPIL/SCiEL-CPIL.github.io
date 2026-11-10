import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # Generate Norm 4 vectors
    candidates_4 = []
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            v[list(combo)] = signs
            candidates_4.append(v)
            
    if not candidates_4:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    V4 = np.array(candidates_4, dtype=np.int64)
    N4 = len(V4)
    
    # Compute conflict adjacency
    # Conflict if dot > 2
    # We store neighbors for fast removal
    conflict_adj = [None] * N4
    
    # Compute dots in chunks or row by row to manage memory/time
    # N4 = 5280. N4^2 = 27M.
    # V4 @ V4.T is fast.
    # But storing 5280 lists of neighbors is fine.
    
    for i in range(N4):
        if time.time() > deadline - 10:
            # Fallback to a small valid set
            return V4[:10]
        
        dots_row = V4[i] @ V4.T
        neighbors = np.where(dots_row > 2)[0]
        # Exclude self? 
        # In greedy, we set active[v]=False, then active[neighbors]=False.
        # If i is in neighbors, it's already False.
        conflict_adj[i] = neighbors

    best_indices = []
    best_k = 0
    
    # Strategy: Random Greedy with Restarts
    # Also try Degree-based ordering
    
    while time.time() < deadline - 5:
        # Random permutation
        order = np.random.permutation(N4)
        
        active_mask = np.ones(N4, dtype=bool)
        current_indices = []
        
        for v in order:
            if not active_mask[v]:
                continue
            current_indices.append(v)
            active_mask[v] = False
            if len(conflict_adj[v]) > 0:
                active_mask[conflict_adj[v]] = False
                
        if len(current_indices) > best_k:
            best_k = len(current_indices)
            best_indices = current_indices.copy()
            
            # Optional: Try local improvement?
            # For now, just keep best.
            
    if best_k == 0:
        # Fallback
        return V4[:1]
        
    return V4[best_indices].astype(np.int64)

if __name__ == "__main__":
    points = kissing_number11()
    print(f"Found {len(points)} points")
