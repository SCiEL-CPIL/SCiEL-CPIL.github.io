import numpy as np
import itertools
import time

def construct():
    D = 11
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Generate Candidates
    # Weight 1
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    v1_arr = np.array(v1_list, dtype=np.int64)
    
    # Weight 4
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    v4_arr = np.array(v4_list, dtype=np.int64)
    N4 = len(v4_arr)
    
    # 2. Precompute Conflicts
    # dots4 = v4_arr @ v4_arr.T
    # This is 5280x5280, ~28M floats. 224 MB. Might be too large/slow to compute fully.
    # Better to compute on the fly or use sparse representation.
    # But for SA, we need fast access.
    # Let's compute the dot products in chunks or just use the property:
    # Conflict if dot > 2.
    
    # To save memory and time, we can compute the conflict matrix once.
    # 5280^2 * 8 bytes = 224 MB. This is acceptable for 120s?
    # Computation time: 5280^3 / 2 flops? No, matrix mult is O(N^3) but optimized.
    # NumPy BLAS should handle 5280x5280 in < 1s.
    
    dots4 = v4_arr @ v4_arr.T
    conflicts4 = dots4 > 2
    np.fill_diagonal(conflicts4, False)
    
    # Convert to list of lists of neighbors for fast iteration
    neighbor_lists = [np.where(conflicts4[i])[0] for i in range(N4)]
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_init():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        # degrees = np.array([len(neighbor_lists[i]) for i in range(N4)])
        # Use min-degree heuristic
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            
            # Find min degree among alive
            # This is slow in Python.
            # Approximation: pick random alive node?
            # Or pre-compute degrees and update?
            # For now, pick random alive node to keep it fast.
            chosen = alive_indices[rng.integers(len(alive_indices))]
            
            in_set[chosen] = True
            # Remove chosen and its neighbors from alive
            to_remove = np.union1d(np.array([chosen]), neighbor_lists[chosen])
            alive[to_remove] = False
            
        return in_set

    def local_search(in_set_init, max_time=10.0):
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        
        T = 1.0
        cooling = 0.999
        min_T = 0.1
        sa_deadline = time.time() + max_time
        
        # Precompute conflict count?
        # conflict_count[i] = number of neighbors of i in in_set
        conflict_count = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            conflict_count[neighbor_lists[i]] += 1
            
        stuck = 0
        while time.time() < sa_deadline:
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0: break
            
            v = not_in[rng.integers(len(not_in))]
            
            # Cost of adding v:
            # Remove all neighbors of v in in_set.
            # Gain: 1 (add v)
            # Loss: number of neighbors in in_set
            nbrs_in_set = neighbor_lists[v][in_set[neighbor_lists[v]]]
            loss = len(nbrs_in_set)
            delta = 1 - loss
            
            if delta >= 0 or rng.random() < np.exp(delta / T):
                # Perform move
                for u in nbrs_in_set:
                    in_set[u] = False
                    # Update conflict counts for neighbors of u
                    # conflict_count[w] decreases by 1 if u was in set
                    # We are removing u, so for all w in neighbors(u), conflict_count[w] -= 1
                    conflict_count[neighbor_lists[u]] -= 1
                
                in_set[v] = True
                # Update conflict counts for neighbors of v
                conflict_count[neighbor_lists[v]] += 1
                
                current_size += delta
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck = 0
                else:
                    stuck += 1
                    
                T *= cooling
                if T < min_T: T = min_T
            else:
                stuck += 1
                T *= cooling
                if T < min_T: T = min_T
                
            if stuck > 500:
                T = max(T, 1.0)
                stuck = 0

        return local_best_in_set, local_best_size

    n_restarts = 0
    max_restarts = 100
    
    while time.time() < deadline - 5 and n_restarts < max_restarts:
        n_restarts += 1
        in_set = greedy_init()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
            
        remaining = deadline - 5 - time.time()
        if remaining < 2: break
        
        sa_time = min(remaining * 0.4, 5.0)
        if sa_time < 1.0: sa_time = 1.0
        
        sa_result, sa_size = local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
            
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = v4_arr[final_v4_indices]
    final_vectors = np.vstack([v1_arr, result_v4])
    
    return final_vectors
