import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate all integer vectors in R^11 with norm squared == 4
    # Types:
    # 1. One coordinate is +/-2, rest 0.
    # 2. Four coordinates are +/-1, rest 0.
    
    candidates = []
    
    # Type 1: +/-2
    for i in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[i] = sign * 2
            candidates.append(v)
            
    # Type 2: Four +/-1
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            
    V = np.array(candidates, dtype=np.int64)
    N = len(V)
    
    # Compute squared distances
    # V is (N, D)
    # dist_sq[i, j] = ||V[i] - V[j]||^2
    # We need dist_sq >= 4 for all i != j in the set.
    # Conflict if dist_sq < 4.
    
    # Efficient distance calculation
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    # Since all have norm^2 = 4:
    # dist_sq = 4 + 4 - 2 a.b = 8 - 2 a.b
    # dist_sq < 4  <=>  8 - 2 a.b < 4  <=>  4 < 2 a.b  <=>  2 < a.b
    # So conflict if dot product > 2.
    
    dots = V @ V.T
    # Conflict if dots[i, j] > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # We want Maximum Independent Set in this graph.
    # Use Randomized Greedy with Local Search.
    
    best_set = np.zeros(N, dtype=bool)
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    def greedy_randomized():
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        
        # Degree of each node in the alive graph
        degrees = conflicts.sum(axis=1).astype(np.int32)
        
        while np.any(alive):
            # Pick a random node from those with minimum degree?
            # Or just random node?
            # Min degree is better for MIS.
            
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            
            # Random tie-breaking
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            
            # Remove chosen and its neighbors from alive
            nbrs = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            # degrees[alive] -= conflicts[alive][:, to_remove].sum(axis=1)
            # This is O(N^2) per step. N=5300.
            # 5300 steps * 5300^2 ops is too slow.
            
            # Optimization:
            # Instead of updating degrees for all, we can just re-calculate min degree on the fly?
            # Or use a heap?
            # Given N=5300, a simple loop might be okay if we vectorize the removal.
            
            # Let's try a different greedy:
            # Just pick random node, remove it and neighbors.
            # This is O(N * AvgDegree).
            
        return in_set

    # The vectorized degree update is:
    # new_degrees = old_degrees - sum of conflicts with removed nodes.
    # We can do:
    # degrees[alive] -= conflicts[alive][:, to_remove].sum(axis=1)
    # This is a matrix-vector multiply. (N_alive x N_removed).
    # Sum of N_removed is roughly N/3?
    # Total cost is sum of (N_i * K_i).
    # Should be manageable for N=5300.
    
    def greedy_fast():
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int32)
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            
            nbrs = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            # Update degrees
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # conflicts[rem] is (N_rem, N)
                # to_remove is (N_rem_to_remove,)
                # We want sum of conflicts[rem][:, to_remove]
                # This is a boolean matrix multiplication
                degrees[rem] -= (conflicts[rem][:, to_remove].sum(axis=1))
                
        return in_set

    # Run multiple times
    for i in range(500):
        if time.time() > deadline - 10:
            break
            
        s = greedy_fast()
        size = s.sum()
        if size > best_size:
            best_size = size
            best_set = s
            
        # Simple local search: try to add nodes
        # (Omitted for brevity in thought, will include in code if time permits, 
        # but Greedy on 5300 nodes with 500 restarts should be sufficient for 165)
        
    result = V[best_set]
    return result
