"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate all vectors with exactly 3 non-zero entries (1 or -1)
    # Number of such vectors: C(11, 3) * 2^3 = 165 * 8 = 1320
    supports = list(itertools.combinations(range(D), 3))
    candidates = []
    for sup in supports:
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            v[sup[0]] = signs[0]
            v[sup[1]] = signs[1]
            v[sup[2]] = signs[2]
            candidates.append(v)
    
    candidates = np.array(candidates)
    n = len(candidates)
    
    # Precompute dot products
    # dot[i, j] = candidates[i] . candidates[j]
    # We need dot[i, j] <= 1 for all i != j
    # Note: dot[i, i] = 3
    
    # For efficiency, we can compute the incompatibility matrix
    # Incompatible if dot > 1
    # Since vectors are sparse, we can compute this fast or just use matrix multiplication
    # 1320x1320 matrix is small enough for numpy
    
    dots = candidates @ candidates.T
    
    # Build adjacency list of incompatibilities
    # incompat[i] is a set of indices j such that dots[i, j] > 1
    # We will use a greedy approach with random restarts
    
    best_set = []
    best_size = 0
    
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle candidates
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        current_set = []
        current_indices = set()
        
        for idx in indices:
            v = candidates[idx]
            # Check compatibility with current_set
            # We can speed this up by maintaining a list of incompatible indices
            # Or just check against current_set directly
            
            # Optimization: if current_set is large, checking against all is O(k)
            # k is at most 1320. 1320 * 1320 = 1.7M ops per restart.
            # 100 restarts = 170M ops. This might be too slow in Python.
            
            # Let's use the precomputed dots matrix.
            # We need to check if any j in current_indices has dots[idx, j] > 1
            
            # To speed up, we can maintain a boolean array `blocked`
            # Initially all False.
            # When we add idx, we mark all j such that dots[idx, j] > 1 as blocked.
            # Then we just check if blocked[idx] is False.
            
            pass # Placeholder for logic below

    # The above loop structure is inefficient if we don't implement the blocking array correctly.
    # Let's implement the blocking array approach.
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        blocked = np.zeros(n, dtype=bool)
        current_set = []
        
        for idx in indices:
            if not blocked[idx]:
                current_set.append(idx)
                # Mark incompatible neighbors as blocked
                # neighbors are those j where dots[idx, j] > 1
                # We can precompute the neighbor list for each idx
                pass

    # Precomputing neighbor lists is better.
    # neighbors[i] = list of j such that dots[i, j] > 1
    
    neighbors = []
    for i in range(n):
        # Find j where dots[i, j] > 1
        # dots[i, :] > 1
        # Exclude i itself (dots[i, i] = 3 > 1)
        mask = dots[i, :] > 1
        mask[i] = False
        neighbors.append(np.where(mask)[0])
    
    best_set_indices = []
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        blocked = np.zeros(n, dtype=bool)
        current_set = []
        
        for idx in indices:
            if not blocked[idx]:
                current_set.append(idx)
                for nb in neighbors[idx]:
                    blocked[nb] = True
        
        if len(current_set) > len(best_set_indices):
            best_set_indices = current_set
            
    # Convert indices to vectors
    result = candidates[best_set_indices]
    
    return result
# EVOLVE-BLOCK-END
