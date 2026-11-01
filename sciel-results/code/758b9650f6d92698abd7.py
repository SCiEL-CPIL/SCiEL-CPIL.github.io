import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # 1. Generate all candidate vectors with norm squared 4
    # Type A: 1 entry +/- 2
    # Type B: 2 entries +/- 2
    # Type C: 4 entries +/- 1
    
    candidates = []
    
    # Type A
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)
            
    # Type B
    for pos1 in range(D):
        for pos2 in range(pos1 + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[pos1] = 2 * s1
                    v[pos2] = 2 * s2
                    candidates.append(v)
                    
    # Type C
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            
    vectors = np.array(candidates, dtype=np.int64)
    N = len(vectors)
    
    # 2. Compute conflict graph
    # Conflict if dot > 2
    # dots = vectors @ vectors.T is N x N, which is 5522 x 5522 ~ 30M entries.
    # This is manageable in memory (30M * 8 bytes = 240MB).
    # Time to compute: 5522^3 / 2 ~ 8.5e10 ops? No, matrix multiplication is optimized.
    # 5500x5500 matmul is fast in NumPy.
    
    dots = vectors @ vectors.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to neighbor lists for faster local search
    # Using lists of arrays or sets?
    # For speed, precompute neighbors as lists of indices.
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    # 3. Heuristic Search for Maximum Independent Set
    
    best_set = np.zeros(N, dtype=bool)
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    def greedy_min_degree():
        """Greedy selection: pick vertex with min degree, remove it and neighbors."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Update degrees for remaining nodes
                # A node u loses degree equal to number of its neighbors in to_remove
                # This is O(N * deg) if done naively.
                # Optimization: precompute conflict matrix row sums?
                # conflicts[rem][:, to_remove].sum(axis=1) is fast with BLAS.
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def local_search(in_set):
        """
        Local search with Add/Remove/Swap moves.
        """
        in_set = in_set.copy()
        # counts[u] = number of neighbors of u in in_set
        counts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            counts[neighbors[i]] += 1
            
        current_size = in_set.sum()
        best_size_ls = current_size
        best_in_set_ls = in_set.copy()
        
        # Greedy fill: add all vertices with count 0
        added = 0
        candidates = np.where((counts == 0) & (~in_set))[0]
        for v in candidates:
            if counts[v] == 0 and not in_set[v]:
                in_set[v] = True
                added += 1
                for u in neighbors[v]:
                    counts[u] += 1
        current_size += added
        if current_size > best_size_ls:
            best_size_ls = current_size
            best_in_set_ls = in_set.copy()
            
        # Simulated Annealing / Random Walk
        # We use a fixed number of steps or time limit
        steps = 20000
        t0 = 1.0
        t_end = 0.01
        alpha = (t_end / t0) ** (1.0 / steps)
        temp = t0
        
        in_set_indices = np.where(in_set)[0]
        not_in_set_indices = np.where(~in_set)[0]
        
        for step in range(steps):
            if time.time() >= deadline - 5:
                break
                
            temp *= alpha
            
            # Try to add a vertex
            if len(not_in_set_indices) > 0 and rng.random() < 0.7:
                v = not_in_set_indices[rng.integers(len(not_in_set_indices))]
                if counts[v] == 0:
                    # Can add
                    in_set[v] = True
                    for u in neighbors[v]:
                        counts[u] += 1
                    current_size += 1
                    # Update indices
                    not_in_set_indices = np.where(~in_set)[0]
                    in_set_indices = np.where(in_set)[0]
                else:
                    # Try to swap: remove one neighbor, add v
                    # Pick a random neighbor w of v that is in_set
                    nbrs_in_set = [u for u in neighbors[v] if in_set[u]]
                    if nbrs_in_set:
                        w = nbrs_in_set[rng.integers(len(nbrs_in_set))]
                        # Remove w, add v
                        # Check if removing w allows adding v (it does, w was the only blocker? No, counts[v] > 0 means multiple blockers potentially)
                        # Actually, if we remove w, counts[v] decreases by 1.
                        # If counts[v] becomes 0, we can add v.
                        # Net change: -1 (remove w) + 1 (add v) = 0.
                        # This is a neutral move.
                        # We accept with probability based on temperature to escape local optima.
                        # But we must ensure v has no OTHER neighbors in set.
                        # If counts[v] == 1, then removing w makes counts[v] 0.
                        # If counts[v] > 1, removing w still leaves counts[v] > 0, so v cannot be added.
                        if counts[v] == 1:
                            # Perform swap
                            in_set[w] = False
                            for u in neighbors[w]:
                                counts[u] -= 1
                            
                            in_set[v] = True
                            for u in neighbors[v]:
                                counts[u] += 1
                            
                            # Size unchanged
                            # Accept with probability 0.5 or temp based
                            if rng.random() < 0.5:
                                # Keep
                                not_in_set_indices = np.where(~in_set)[0]
                                in_set_indices = np.where(in_set)[0]
                            else:
                                # Revert
                                in_set[v] = False
                                for u in neighbors[v]:
                                    counts[u] -= 1
                                in_set[w] = True
                                for u in neighbors[w]:
                                    counts[u] += 1
                                
            else:
                # Try to remove a vertex to allow others? 
                # Or just random walk.
                if len(in_set_indices) > 0:
                    v = in_set_indices[rng.integers(len(in_set_indices))]
                    in_set[v] = False
                    for u in neighbors[v]:
                        counts[u] -= 1
                    current_size -= 1
                    
                    # Try to add back or other vertices
                    # Greedy fill again? Too expensive.
                    # Just accept removal with low prob to escape.
                    if rng.random() > np.exp(-1/temp):
                        # Revert
                        in_set[v] = True
                        for u in neighbors[v]:
                            counts[u] += 1
                        current_size += 1
                    else:
                        not_in_set_indices = np.where(~in_set)[0]
                        in_set_indices = np.where(in_set)[0]
                        
            if current_size > best_size_ls:
                best_size_ls = current_size
                best_in_set_ls = in_set.copy()
                
        return best_in_set_ls

    # Main Loop
    n_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Greedy Init
        in_set = greedy_min_degree()
        
        # Local Search
        in_set = local_search(in_set)
        
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_set = in_set.copy()
            
    final_indices = np.where(best_set)[0]
    result = vectors[final_indices]
    
    return result
