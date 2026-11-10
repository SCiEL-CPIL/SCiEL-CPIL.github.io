import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Find a large set of integer points in R^11 such that:
    - All points have integer coordinates
    - No point is the zero vector
    - For all pairs (a, b): ||a - b||^2 >= max(||p||^2 for p in points)
    
    This is equivalent to: for all pairs, ||a||^2 + ||b||^2 - 2<a,b> >= max_norm
    
    Strategy: Use norm-2 vectors (squared norm = 2). There are C(11,2)*4 = 220 such vectors.
    For two norm-2 vectors a, b: ||a-b||^2 = 2 + 2 - 2<a,b> = 4 - 2<a,b>
    We need 4 - 2<a,b> >= 2, i.e., <a,b> <= 1.
    
    Two norm-2 vectors with <a,b> = 1 are "conflicting" (can't both be in set).
    <a,b> = 1 happens when they share exactly one coordinate with same sign.
    
    We need a maximum independent set in the conflict graph where vertices are the 220 norm-2
    vectors and edges connect conflicting pairs.
    
    Alternatively, we can use a greedy approach with randomization to find a large independent set.
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 500  # Leave 100s margin before 600s limit
    
    # Generate all norm-2 integer vectors
    vectors = []
    for i in range(d):
        for j in range(i + 1, d):
            vectors.append(np.array([0]*i + [1] + [0]*(j-i-1) + [1] + [0]*(d-j-1), dtype=np.int64))
            vectors.append(np.array([0]*i + [1] + [0]*(j-i-1) + [-1] + [0]*(d-j-1), dtype=np.int64))
            vectors.append(np.array([0]*i + [-1] + [0]*(j-i-1) + [1] + [0]*(d-j-1), dtype=np.int64))
            vectors.append(np.array([0]*i + [-1] + [0]*(j-i-1) + [-1] + [0]*(d-j-1), dtype=np.int64))
    
    n = len(vectors)  # 220
    vecs = np.array(vectors)
    
    # Compute conflict matrix: two vectors conflict if their dot product > 0 (i.e., >= 1)
    # For norm-2 vectors, dot product can be -2, -1, 0, 1, 2
    # <a,b> = 2 means a == b (same vector)
    # <a,b> = 1 means they share one coordinate with same sign -> conflict
    # <a,b> = 0 means no conflict
    # <a,b> = -1 means no conflict (distance = 4-2*(-1) = 6 >= 2)
    # <a,b> = -2 means a = -b, no conflict
    
    dot_products = vecs @ vecs.T
    # Conflict if dot product >= 1 (and not the same vector)
    conflict = (dot_products >= 1).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # Greedy independent set with multiple random orderings
    best_set = []
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Try many random orderings
    for trial in range(2000):
        if time.time() > deadline:
            break
        
        order = rng.permutation(n)
        selected = np.zeros(n, dtype=bool)
        
        for idx in order:
            # Check if idx conflicts with any already selected
            if np.any(conflict[idx, selected]):
                continue
            selected[idx] = True
        
        size = np.sum(selected)
        if size > best_size:
            best_size = size
            best_set = np.where(selected)[0].tolist()
    
    # Local improvement: try to add more or swap
    # Simple local search: try adding unselected vertices
    for pass_num in range(10):
        if time.time() > deadline:
            break
        improved = False
        selected_mask = np.zeros(n, dtype=bool)
        for idx in best_set:
            selected_mask[idx] = True
        
        # Try to add each unselected vertex
        for idx in range(n):
            if selected_mask[idx]:
                continue
            if not np.any(conflict[idx, selected_mask]):
                selected_mask[idx] = True
                improved = True
        
        if improved:
            best_set = np.where(selected_mask)[0].tolist()
            best_size = len(best_set)
    
    # Convert to point array
    if best_set:
        result = vecs[best_set]
    else:
        # Fallback: just use all 220 (should not happen)
        result = vecs
    
    return np.around(result).astype(np.int64)
