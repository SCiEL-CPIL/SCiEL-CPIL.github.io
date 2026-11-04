import numpy as np
import itertools
import time
import random
from collections import defaultdict

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Generate candidate vectors
    # Type 1: 4 positions with ±1 (norm 2)
    # Type 2: 2 positions with ±2 (norm 2*sqrt(2))
    # Type 3: 3 positions with ±2 (norm 2*sqrt(3))
    # Type 4: 1 position with ±2 (norm 2)
    # Type 5: 4 positions with (±1, ±1, ±1, ±1) - same as Type 1
    
    candidates = []
    
    # Type 1: 4-subsets, ±1 entries
    for subset in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for i, pos in enumerate(subset):
                v[pos] = signs[i]
            candidates.append(v)
    
    # Type 2: 2-subsets, ±2 entries
    for subset in itertools.combinations(range(D), 2):
        for signs in itertools.product([2, -2], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            for i, pos in enumerate(subset):
                v[pos] = signs[i]
            candidates.append(v)
    
    # Type 3: 3-subsets, ±2 entries
    for subset in itertools.combinations(range(D), 3):
        for signs in itertools.product([2, -2], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for i, pos in enumerate(subset):
                v[pos] = signs[i]
            candidates.append(v)
    
    # Type 4: 1-subset, ±2 entries
    for pos in range(D):
        for sign in [2, -2]:
            v = np.zeros(D, dtype=np.int64)
            v[pos] = sign
            candidates.append(v)
    
    n = len(candidates)
    # n = C(11,4)*16 + C(11,2)*4 + C(11,3)*8 + 11*2
    # = 330*16 + 55*4 + 165*8 + 22
    # = 5280 + 220 + 1320 + 22 = 6842
    
    # Precompute norms squared
    norms_sq = np.array([np.dot(v, v) for v in candidates], dtype=np.int64)
    
    # Precompute pairwise distances squared
    # dist_sq[i,j] = ||v_i - v_j||^2
    # We need dist_sq[i,j] >= max(norms_sq[i], norms_sq[j])
    
    # Build adjacency: i and j are incompatible if dist_sq < max(norms_sq[i], norms_sq[j])
    # For memory efficiency, store as bitsets or use a different approach
    
    # Since n can be ~6842, pairwise is ~23M pairs. Let's compute on the fly or use numpy.
    
    # Actually, let's use a more memory-efficient approach.
    # Convert candidates to a 2D array
    V = np.array(candidates, dtype=np.int64)
    
    # Compute all pairwise distances in chunks to avoid memory issues
    # dist_sq[i,j] = ||V[i]||^2 + ||V[j]||^2 - 2*V[i]·V[j]
    
    # Precompute dot products
    # This is V @ V.T, which is n x n
    # n ~ 6842, so 6842^2 ~ 47M entries, ~376MB as int64. Might be tight.
    
    # Let's do it in chunks
    chunk_size = 500
    incompatible = [set() for _ in range(n)]
    
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        V_chunk = V[i_start:i_end]
        # Compute dot products of V_chunk with all V
        dots = V_chunk @ V.T  # shape (chunk, n)
        
        for local_i in range(i_end - i_start):
            gi = i_start + local_i
            norm_i = norms_sq[gi]
            for j in range(n):
                if gi == j:
                    continue
                dist_sq = norm_i + norms_sq[j] - 2 * dots[local_i, j]
                if dist_sq < max(norm_i, norms_sq[j]):
                    incompatible[gi].add(j)
                    incompatible[j].add(gi)
        
        if time.time() > deadline - 30:
            break
    
    # Now find a large independent set
    # Use greedy + local search
    
    def greedy_min_degree(rng):
        available = set(range(n))
        is_set = set()
        while available:
            min_deg = float('inf')
            candidates_list = []
            for v in available:
                deg = len(incompatible[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    candidates_list = [v]
                elif deg == min_deg:
                    candidates_list.append(v)
            if not candidates_list:
                break
            best_node = rng.choice(candidates_list)
            is_set.add(best_node)
            to_remove = incompatible[best_node] | {best_node}
            available -= to_remove
        return is_set
    
    def greedy_random_order(rng):
        order = list(range(n))
        rng.shuffle(order)
        is_set = set()
        for v in order:
            if not (incompatible[v] & is_set):
                is_set.add(v)
        return is_set
    
    def local_search_add(is_set, rng):
        is_set = set(is_set)
        improved = True
        while improved:
            improved = False
            candidates = [v for v in range(n) if v not in is_set and not (incompatible[v] & is_set)]
            if candidates:
                rng.shuffle(candidates)
                for v in candidates:
                    if v not in is_set and not (incompatible[v] & is_set):
                        is_set.add(v)
                        improved = True
        return is_set
    
    def local_search_swap(is_set, rng, max_iter=5):
        is_set = set(is_set)
        current_size = len(is_set)
        
        for _ in range(max_iter):
            improved = False
            for v in list(is_set):
                if time.time() > deadline:
                    return is_set
                trial = is_set - {v}
                # Try to add all compatible nodes not in trial
                added = 0
                for u in range(n):
                    if u not in trial and not (incompatible[u] & trial):
                        trial.add(u)
                        added += 1
                
                if len(trial) > current_size:
                    is_set = trial
                    current_size = len(is_set)
                    improved = True
                    break
            
            if not improved:
                break
        
        return is_set
    
    best_is = set()
    best_size = 0
    
    seed_val = 0
    while time.time() < deadline - 10:
        rng = random.Random(seed_val)
        
        if seed_val % 2 == 0:
            is_set = greedy_min_degree(rng)
        else:
            is_set = greedy_random_order(rng)
        
        is_set = local_search_add(is_set, rng)
        
        if time.time() < deadline - 15:
            is_set = local_search_swap(is_set, rng, max_iter=3)
        
        if len(is_set) > best_size:
            best_size = len(is_set)
            best_is = is_set
        
        seed_val += 1
    
    # Construct vectors
    vectors = []
    for idx in best_is:
        vectors.append(candidates[idx])
    
    if not vectors:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        vectors.append(v)
        
    return np.array(vectors, dtype=np.int64)
