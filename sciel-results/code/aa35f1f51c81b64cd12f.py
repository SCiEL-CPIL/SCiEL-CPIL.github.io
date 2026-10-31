import numpy as np
import itertools
import time

D = 11

def generate_vectors(weights):
    """Generate all integer vectors with given weights (non-zero entries are +/-1)."""
    vectors = []
    for w in weights:
        for combo in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(combo, signs):
                    v[idx] = s
                vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Use weight 3 and 4 vectors
    candidates = generate_vectors([3, 4])
    n = len(candidates)
    
    if n == 0:
        return np.empty((0, D), dtype=np.int64)
    
    # Precompute dot products
    # This might be memory intensive: 6600 x 6600 int64 is ~350 MB.
    # Let's use int16 or int32 if possible, or compute in chunks.
    # Max dot product for weight 4 is 4. Weight 3 is 3.
    # So int8 is sufficient.
    dots = candidates @ candidates.T
    
    # Determine conflicts
    # We need to know the weight of each vector to apply the correct threshold.
    weights = np.sum(candidates != 0, axis=1)
    
    # Thresholds:
    # w=4, w=4: dot > 2
    # w=3, w=3: dot > 1
    # w=3, w=4: dot > 1
    
    # Create a threshold matrix
    # If weight[i] == 4 and weight[j] == 4: thresh = 2
    # Else: thresh = 1
    
    # We can compute this efficiently
    # thresh_matrix[i,j] = 2 if w_i==4 and w_j==4 else 1
    
    # Using broadcasting
    w4_mask = (weights == 4)
    # w4_mask is shape (n,)
    # thresh_matrix = np.where(np.outer(w4_mask, w4_mask), 2, 1)
    
    # This creates an (n,n) array.
    # Conflict if dots[i,j] > thresh_matrix[i,j]
    
    # To save memory, we can compute conflict in chunks or just use the condition directly.
    # But we need a boolean matrix for the greedy step.
    
    # Let's create the threshold matrix
    thresh = np.ones((n, n), dtype=np.int8)
    # Indices where both are weight 4
    # We can use a mask
    mask_44 = np.outer(w4_mask, w4_mask)
    thresh[mask_44] = 2
    
    conflict = dots > thresh
    
    # Remove self-loops
    np.fill_diagonal(conflict, False)
    
    # Greedy MIS with many restarts
    best_size = 0
    best_result = np.empty((0, D), dtype=np.int64)
    
    # Number of restarts depends on speed
    # Estimate: N=6600. Neighbors ~ 50-100.
    # One restart ~ 6600 * 100 = 660k ops. ~ 1ms?
    # So we can do 1000 restarts in 1s.
    # Let's run until deadline.
    
    seed = 0
    while time.time() < deadline:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        blocked = np.zeros(n, dtype=bool)
        selected = []
        
        for idx in order:
            if blocked[idx]:
                continue
            selected.append(idx)
            blocked[idx] = True
            blocked[conflict[idx]] = True
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = candidates[np.array(selected, dtype=np.int64)]
            
        seed += 1
        
    return best_result
