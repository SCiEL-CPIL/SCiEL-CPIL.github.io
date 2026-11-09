import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe (kill at 150, buffer 40)
    
    # 1. Generate Weight-4 vectors
    v4_list = []
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            v4_list.append(v)
    V4 = np.array(v4_list, dtype=np.int8)
    
    V = V4
    n_vecs = V.shape[0]
    
    # Compute dot products
    dots = V @ V.T
    
    # Conflict if dot > 2
    # We want MIS in graph where edge exists if dots[i,j] > 2.
    # Note: diagonal is 4, so self is conflict.
    
    best_size = 0
    best_indices = []
    
    # Pre-compute neighbor sets for fast set operations
    # neighbors[i] is a set of indices j such that conflict_masks[i,j] is True
    # conflict_masks[i,j] is True if dots[i,j] > 2
    
    # Using sets is faster for "remove neighbors" than bitsets in Python for this size
    # N=5280, Degree ~ 56.
    
    neighbors = [set() for _ in range(n_vecs)]
    for i in range(n_vecs):
        # Find all j such that dots[i,j] > 2
        # Vectorized: np.nonzero(dots[i] > 2)[0]
        nbrs = np.nonzero(dots[i] > 2)[0]
        neighbors[i] = set(nbrs.tolist())
        
    # Random Order Greedy with Restarts
    # This is O(N+E) per restart, which is very fast.
    
    num_restarts = 5000 # High number, will be limited by time
    
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Randomize the order of consideration
        # Using a permutation
        order = np.random.permutation(n_vecs)
        
        available = set(range(n_vecs))
        selected = []
        
        for u in order:
            if u in available:
                selected.append(u)
                # Remove u and its neighbors from available
                # u is in neighbors[u] because dots[u,u]=4 > 2
                available.difference_update(neighbors[u])
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_indices = selected
            # Optional: break if we reach a very high number, but let's try to maximize
            # Theoretical max is unknown, but 3500+ is good.
            if best_size > 4000:
                break

    if best_size == 0:
        return V.astype(np.int64)

    return V[best_indices].astype(np.int64)
