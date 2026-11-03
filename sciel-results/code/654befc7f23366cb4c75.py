import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Generate all weight 3 vectors
    vectors = []
    for support in itertools.combinations(range(D), 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[support[0]] = s1
                    v[support[1]] = s2
                    v[support[2]] = s3
                    vectors.append(np.array(v, dtype=np.int64))
    
    vectors = np.array(vectors)
    N = len(vectors)
    
    # Precompute norms squared (all are 3)
    # We need min pairwise distance squared >= 3
    
    best_count = 0
    best_indices = []
    
    # Greedy with random restarts
    rng = np.random.default_rng(42)
    
    # Precompute a fast conflict check?
    # Distance squared between u and v is ||u||^2 + ||v||^2 - 2 u.v = 6 - 2 u.v
    # We need 6 - 2 u.v >= 3 => 2 u.v <= 3 => u.v <= 1.5 => u.v <= 1 (integer)
    # So the condition is: for any two selected vectors u, v, their dot product u.v <= 1.
    
    # This is a much simpler condition!
    # u.v is integer.
    # If supports disjoint, u.v = 0 <= 1. OK.
    # If intersect 1, u.v = +/- 1 <= 1. OK.
    # If intersect 2, u.v = sum of 2 products.
    #   If signs agree on both: 1+1=2 > 1. FAIL.
    #   If signs agree on one, disagree on one: 1-1=0 <= 1. OK.
    #   If signs disagree on both: -1-1=-2 <= 1. OK.
    # If intersect 3 (same support), u.v = sum of 3 products.
    #   If same: 3 > 1. FAIL (but u=v, dist=0).
    #   If differ 1: 1+1-1=1 <= 1. OK.
    #   If differ 2: 1-1-1=-1 <= 1. OK.
    #   If differ 3: -1-1-1=-3 <= 1. OK.
    
    # So the condition is simply: for all distinct u, v in the set, u.v <= 1.
    
    # We can precompute the dot product matrix M where M[i,j] = vectors[i].vectors[j]
    # Then we need to find the largest subset of indices such that M[i,j] <= 1 for all i!=j.
    # This is a maximum independent set in the graph where edge exists if M[i,j] > 1.
    
    # Calculate M
    # M = vectors @ vectors.T
    # This is 1320 x 1320 matrix. Fast.
    
    M = vectors @ vectors.T
    # Conflict if M[i,j] > 1 and i != j
    # Note: M[i,i] = 3 > 1, but we don't check i==i.
    
    # Greedy approach
    # We can use a simple heuristic:
    # 1. Sort vectors by degree in conflict graph (number of conflicts) ascending?
    #    Or just random shuffle.
    
    # Let's try a few strategies.
    
    for run in range(1000):
        if time.time() > deadline:
            break
            
        # Random permutation
        perm = rng.permutation(N)
        
        selected = []
        # To speed up, we can maintain a set of "forbidden" indices?
        # Or just check against selected.
        
        # Since N=1320, checking against selected list is O(K).
        # Total O(N*K).
        
        # Optimization:
        # If we maintain a boolean array 'conflict_with_selected' of size N,
        # initially all False.
        # When we select vector idx, we mark all j where M[idx, j] > 1 as True.
        # Then we just check if conflict_with_selected[perm[k]] is False.
        
        forbidden = np.zeros(N, dtype=bool)
        
        for idx in perm:
            if forbidden[idx]:
                continue
            
            # Check if we can add idx
            # Actually, the 'forbidden' array already tells us if it conflicts with ANY selected vector.
            # So if not forbidden[idx], we can add it.
            
            selected.append(idx)
            
            # Update forbidden
            # forbidden |= (M[idx, :] > 1)
            # But we must not forbid idx itself? It doesn't matter, we won't pick it again.
            forbidden |= (M[idx, :] > 1)
            
        if len(selected) > best_count:
            best_count = len(selected)
            best_indices = selected[:]
            
    # Construct the result
    if best_count > 0:
        result = vectors[best_indices]
    else:
        # Fallback to seed solution if something went wrong
        vectors_seed = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors_seed.append(v)
        result = np.array(vectors_seed, dtype=np.int64)
        
    return result
