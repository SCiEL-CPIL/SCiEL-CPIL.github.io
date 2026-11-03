import numpy as np
import time
import random
import itertools

D = 11

def get_vectors_of_weight(k):
    vecs = []
    # Generate combinations of k indices
    for comb in itertools.combinations(range(D), k):
        # Generate all sign combinations
        for signs in itertools.product([1, -1], repeat=k):
            v = [0] * D
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs.append(v)
    return np.array(vecs, dtype=np.int8)

def solve_independent_set(V, threshold, time_limit=10.0):
    n = len(V)
    if n == 0:
        return np.array([], dtype=np.int8).reshape(0, D)
    
    dots = V @ V.T
    # Conflict if dots[i][j] > threshold
    # Note: diagonal is k. If k > threshold, self-conflict?
    # We don't select same vector twice.
    # We only care about i != j.
    
    # Initial degrees
    # deg[i] = count of j != s.t. dots[i][j] > threshold
    # dots[i][i] = k. If k > threshold, it counts.
    # We subtract 1 for self if k > threshold.
    
    conflict_matrix = dots > threshold
    # Ensure diagonal is False for degree calculation?
    # np.fill_diagonal(conflict_matrix, False)
    # But we need to know neighbors.
    
    deg = conflict_matrix.sum(axis=1)
    np.fill_diagonal(conflict_matrix, False) # For neighbor finding
    # Recalculate deg correctly?
    # deg[i] was count of True in row i.
    # If conflict_matrix[i][i] was True, deg[i] included self.
    # We want degree in simple graph (no self loops).
    # So if k > threshold, deg[i] -= 1.
    
    if V[0] @ V[0] > threshold:
        deg -= 1
        
    best_set = set()
    start_time = time.time()
    
    # Multiple restarts
    while time.time() - start_time < time_limit:
        # Randomize degrees slightly to break symmetry?
        # Or just run min-degree greedy.
        
        available = np.ones(n, dtype=bool)
        selected = []
        current_deg = deg.copy()
        
        while available.any():
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            
            # Pick min degree
            # To add randomness, pick randomly from min-degree nodes
            min_deg = np.min(current_deg[avail_indices])
            min_indices = avail_indices[current_deg[avail_indices] == min_deg]
            idx = min_indices[random.randint(0, len(min_indices)-1)]
            
            selected.append(idx)
            available[idx] = False
            
            # Update neighbors
            neighbors = np.where(conflict_matrix[idx])[0]
            available[neighbors] = False
            
            # Update degrees
            # For each neighbor j, current_deg[j] -= 1
            # Only if still available? No, degree is property of graph.
            # But for greedy, we only care about available nodes.
            # Updating all neighbors is fine.
            
            # Vectorized update is hard in loop.
            # n is small, loop is okay.
            for j in neighbors:
                if available[j]: # Optimization: only update if still available
                    current_deg[j] -= 1
                # Actually, if j is removed, we don't care about its degree.
                # But current_deg is used for min selection among available.
                # So we only need accurate degrees for available nodes.
            
        if len(selected) > len(best_set):
            best_set = set(selected)
            
    if not best_set:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    return V[list(best_set)]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    best_result = None
    best_size = 0
    
    # Try weight 3
    t_start = time.time()
    V3 = get_vectors_of_weight(3)
    t_gen = time.time() - t_start
    
    # Time left for solving
    time_left = deadline - (time.time() - start_time) - 5 # 5s margin
    
    if time_left > 0:
        # Split time between k=3 and k=4?
        # Or try k=3 first, then k=4 if time remains.
        # k=4 has more vectors, might take longer to solve.
        
        # Let's try k=3 with 40% of time, k=4 with 60%?
        # Or just run k=3 for 20s, k=4 for 20s.
        
        time_alloc_3 = min(20.0, time_left * 0.4)
        time_alloc_4 = min(20.0, time_left * 0.6)
        
        if time_alloc_3 > 1:
            res3 = solve_independent_set(V3, 1, time_limit=time_alloc_3)
            if len(res3) > best_size:
                best_size = len(res3)
                best_result = res3
                
        # Check time
        elapsed = time.time() - start_time
        if elapsed < deadline - 5:
            V4 = get_vectors_of_weight(4)
            time_alloc_4 = min(20.0, deadline - elapsed - 5)
            if time_alloc_4 > 1:
                res4 = solve_independent_set(V4, 2, time_limit=time_alloc_4)
                if len(res4) > best_size:
                    best_size = len(res4)
                    best_result = res4
                    
    if best_result is None:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return best_result
