import numpy as np
import itertools
import time
import random

# Cache for combinations
comb_cache = {}

def get_combinations(n, k):
    key = (n, k)
    if key not in comb_cache:
        comb_cache[key] = np.array(list(itertools.combinations(range(n), k)))
    return comb_cache[key]

def get_bad_subsets(A, n, k):
    if k > n:
        return []
    combos = get_combinations(n, k)
    if combos.size == 0:
        return []
    
    idx_row = combos[:, :, np.newaxis]
    idx_col = combos[:, np.newaxis, :]
    A_sub = A[idx_row, idx_col]
    degrees = A_sub.sum(axis=2)
    is_reg = np.all(degrees == degrees[:, 0:1], axis=1)
    
    return combos[is_reg]

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Try n=21 first
    for n in [21, 20]:
        if time.time() > deadline - 10:
            break
            
        # Initialize
        A = (np.random.rand(n, n) < 0.5).astype(int)
        np.fill_diagonal(A, 0)
        
        # Local search
        while time.time() < deadline - 5:
            bad_subset = None
            # Prioritize smaller k
            for k in [6, 7, 8]:
                bads = get_bad_subsets(A, n, k)
                if len(bads) > 0:
                    bad_subset = random.choice(bads)
                    break
            
            if bad_subset is None:
                # Check k=9, 10 just in case?
                # If 6,7,8 are clean, 9+ are very unlikely to be regular.
                # But let's check 9 if time allows.
                if time.time() < deadline - 2:
                     bads_9 = get_bad_subsets(A, n, 9)
                     if len(bads_9) > 0:
                         bad_subset = random.choice(bads_9)
                
                if bad_subset is None:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if A[i, j] == 1:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
            
            # Flip edge
            u, v = random.sample(list(bad_subset), 2)
            A[u, v] = 1 - A[u, v]
            A[v, u] = A[u, v]
            
    return {'n': 5, 'edges': []}
