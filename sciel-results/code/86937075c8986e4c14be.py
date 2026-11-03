import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s margin
    
    # Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)
    
    # Precompute intersection sizes
    # intersection[i][j] = len(set(supports[i]) & set(supports[j]))
    # We need intersection <= 2. So conflict if >= 3.
    # 330x330 matrix is small.
    conflict = np.zeros((n_supports, n_supports), dtype=bool)
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i + 1, n_supports):
            sj = set(supports[j])
            inter = len(si.intersection(sj))
            if inter >= 3:
                conflict[i][j] = True
                conflict[j][i] = True

    best_mis = []
    
    # Greedy MIS with random restarts
    num_restarts = 1000
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random order
        order = np.random.permutation(n_supports)
        current_mis = []
        in_mis = np.zeros(n_supports, dtype=bool)
        
        for node in order:
            # Check if node conflicts with any in current_mis
            # conflict[node] & in_mis
            if not np.any(conflict[node] & in_mis):
                current_mis.append(node)
                in_mis[node] = True
        
        if len(current_mis) > len(best_mis):
            best_mis = current_mis
            if len(best_mis) > 40: # Heuristic: if we find a large one, maybe stop early?
                 # But let's run a few more to be sure.
                 pass

    # Construct vectors
    vectors = []
    for s_idx in best_mis:
        support = supports[s_idx]
        # Generate all 16 sign variants
        # Indices in support are 4.
        # Signs are +/-1 for each.
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for k, idx in enumerate(support):
                v[idx] = signs[k]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
