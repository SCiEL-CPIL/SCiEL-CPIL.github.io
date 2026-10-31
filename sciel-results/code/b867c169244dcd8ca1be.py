import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11
    # Norm^2 = 4
    pos_sets = list(itertools.combinations(range(D), 4))
    signs_list = list(itertools.product([1, -1], repeat=4))
    
    vectors = []
    for p in pos_sets:
        for s in signs_list:
            v = [0] * D
            for pos, sig in zip(p, s):
                v[pos] = sig
            vectors.append(v)
            
    N = len(vectors)
    V = np.array(vectors, dtype=np.int8)
    
    # Generate weight-1 vectors of type (2, 0, ...)
    # Norm^2 = 4
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    V1 = np.array(v1_candidates, dtype=np.int8)
    
    # Check compatibility of V1 with V
    # V1 dot V:
    # If V has 1 at pos, V1 has 2 at pos -> dot = 2. OK (<=2)
    # If V has -1 at pos, V1 has 2 at pos -> dot = -2. OK
    # If V has 0 at pos, dot = 0. OK
    # So all V1 are compatible with all V.
    # V1 with V1:
    # Same pos, same sign: dot = 4. Conflict!
    # Same pos, diff sign: dot = -4. OK
    # Diff pos: dot = 0. OK
    # So we can pick at most one from each pair (v, -v) of V1?
    # Actually, we can pick all V1 except we can't pick both v and -v for same pos.
    # So max 11 from V1.
    # But wait, if we pick v and -v, dist is 4. Norm is 2. 4 >= 2. OK.
    # Wait, condition is dist >= max norm.
    # Max norm is 2.
    # Dist between (2,0..) and (-2,0..) is 4. 4 >= 2. OK.
    # So we can take ALL 22 V1 vectors?
    # Let's re-verify dot product condition for same norm vectors.
    # |u-v|^2 = 4+4 - 2u.v = 8 - 2u.v.
    # We need |u-v| >= 2 => |u-v|^2 >= 4.
    # 8 - 2u.v >= 4 => 4 >= 2u.v => u.v <= 2.
    # For v=(2,0..) and -v=(-2,0..), u.v = -4. -4 <= 2. OK.
    # So yes, all 22 V1 are mutually compatible.
    
    # Now find MIS in V (5280 nodes).
    # Conflict if u.v > 2.
    
    # Compute dot products
    # V is (5280, 11)
    # Dot matrix is (5280, 5280)
    # This is 5280^2 * 8 bytes = 223 MB. Manageable.
    # But computing it takes time.
    # We can do it in chunks or use float32 to save memory/time.
    
    # Use float32 for speed and memory
    Vf = V.astype(np.float32)
    
    # We only need the upper triangle or just the conflict mask.
    # To save time, we can compute row by row and check conflicts.
    # But vectorized full matrix is faster if it fits in cache.
    # 5280x5280 float32 is ~111 MB. Fits in L3 cache? Maybe not.
    # Let's do it in blocks to be safe and fast.
    
    # Actually, we can just compute the full dot product matrix.
    # It's a single BLAS call.
    try:
        dot_prod = Vf @ Vf.T
    except MemoryError:
        # Fallback to chunked
        dot_prod = np.zeros((N, N), dtype=np.float32)
        chunk = 1000
        for i in range(0, N, chunk):
            end = min(i + chunk, N)
            dot_prod[i:end] = Vf[i:end] @ Vf.T
            
    # Conflict mask: dot_prod > 2
    # Note: diagonal is 4. We don't care about self-conflict.
    # We want to select a set of indices such that no two selected have conflict.
    
    # Greedy MIS with randomization
    best_set = []
    best_size = 0
    
    # We will run several restarts with different random orders
    # Or use a deterministic greedy with min-degree, which is usually good.
    
    # Let's use a vectorized greedy:
    # 1. Compute degrees (number of conflicts).
    # 2. Pick node with min degree.
    # 3. Remove it and its neighbors from consideration.
    # 4. Repeat.
    
    # Degrees: sum of conflict mask (excluding diagonal)
    # conflict_mask = dot_prod > 2
    # diag is 4 > 2, so it's True.
    # We want to exclude self.
    
    # To speed up, we can just use a heuristic:
    # Randomly shuffle indices, then greedily pick if compatible with all previously picked.
    # This is O(K^2) per restart. K~512. 512^2 = 260k ops. Very fast.
    # We can do many restarts.
    
    indices = np.arange(N)
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Randomize order
        random.shuffle(indices)
        
        selected = []
        selected_vecs = []
        
        # To make it faster, we can check compatibility against the selected set.
        # Since selected set grows, this is O(K^2).
        # We can vectorize the check against the current selected set.
        
        # Pre-compute dot products of all V with all V? Already have dot_prod.
        # For a candidate i, we need dot_prod[i, j] <= 2 for all j in selected.
        
        # We can maintain a boolean mask of "compatible with all selected so far".
        # Initially all True.
        # When we select i, we update mask: mask &= (dot_prod[:, i] <= 2)
        # Then pick next i from where mask is True.
        
        compatible = np.ones(N, dtype=bool)
        
        # To speed up selection, we can pick randomly from compatible, or min-degree.
        # Min-degree in the remaining graph is hard to maintain dynamically.
        # Random selection from compatible is fast.
        
        # Let's try random selection from compatible.
        # To improve, we can bias towards nodes with fewer conflicts overall.
        # Precompute degree: deg = (dot_prod > 2).sum(axis=1)
        deg = (dot_prod > 2).sum(axis=1)
        
        # Sort indices by degree? No, we shuffle.
        # We can weight the random choice by 1/deg?
        
        # Simple random choice from compatible:
        # np.where(compatible)[0]
        
        # Optimization:
        # Instead of iterating through all N to find a compatible one,
        # we can just iterate through the shuffled indices and pick the first compatible one?
        # No, that's not greedy.
        
        # Let's just iterate through shuffled indices.
        # If compatible[i] is True, select i.
        # Update compatible.
        
        # This is O(N * K) per restart? No, O(N) to iterate, O(K) to update.
        # Total O(N*K). 5280 * 512 ~ 2.7M ops. Fast.
        
        for i in indices:
            if compatible[i]:
                selected.append(i)
                # Update compatible
                # compatible = compatible & (dot_prod[:, i] <= 2)
                # This is a vectorized operation on 5280 elements.
                compatible &= (dot_prod[:, i] <= 2)
                
        size = len(selected)
        if size > best_size:
            best_size = size
            best_set = selected[:]
            
        # Early stopping if we hit 512
        if best_size >= 512:
            break
            
    # Construct result
    if best_set:
        result_v4 = V[best_set]
    else:
        result_v4 = V[:1] # Fallback
        
    final_vectors = np.vstack([result_v4, V1])
    
    return final_vectors
