import numpy as np
import time
import random
from itertools import combinations

def generate_weight_vectors(d, weight):
    """
    Generates all integer vectors of length d with exactly 'weight' non-zero entries,
    each being +/- 1.
    Returns a list of numpy arrays.
    """
    vectors = []
    indices = list(range(d))
    
    # Generate all combinations of indices
    for combo in combinations(indices, weight):
        # Generate all sign combinations (2^weight)
        # Use bit manipulation for speed
        for mask in range(2 ** weight):
            v = np.zeros(d, dtype=np.int64)
            for k, idx in enumerate(combo):
                if (mask >> k) & 1:
                    v[idx] = 1
                else:
                    v[idx] = -1
            vectors.append(v)
            
    return vectors

def greedy_select(candidates, max_dot_product, time_limit=50):
    """
    Greedily selects a subset of candidates such that the dot product
    between any two selected vectors is <= max_dot_product.
    Returns the selected vectors as a numpy array.
    """
    if not candidates:
        return np.array([], dtype=np.int64).reshape(0, 11)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    best_selected = []
    best_size = 0
    start_time = time.time()
    
    # Try multiple random orderings
    for attempt in range(100):
        if time.time() - start_time > time_limit:
            break
            
        indices = np.random.permutation(n)
        selected = []
        selected_matrix = None
        
        for idx in indices:
            v = candidates[idx]
            
            if not selected:
                selected.append(v)
                selected_matrix = v.reshape(1, -1)
                continue
            
            # Check compatibility with all selected
            # Compute dot products
            dots = np.dot(selected_matrix, v)
            
            if np.max(dots) <= max_dot_product:
                selected.append(v)
                selected_matrix = np.vstack([selected_matrix, v.reshape(1, -1)])
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
            
        # Early exit if we found a very good solution?
        # Or if we are running out of time.
        
    return np.array(best_selected, dtype=np.int64) if best_selected else np.array([], dtype=np.int64).reshape(0, 11)

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 600 - 10  # Safety margin
    
    # Strategy 1: Weight 2 vectors (Norm^2 = 2)
    # Condition: dist^2 >= 2 => 4 - 2*u.v >= 2 => u.v <= 1
    # We know the full set of 220 is valid.
    w2_vectors = generate_weight_vectors(d, 2)
    w2_array = np.array(w2_vectors, dtype=np.int64)
    
    # Verify w2_array validity (should be true)
    # For performance, we assume it's valid based on previous analysis.
    # But let's do a quick check if we have time? No, it's fast enough to just trust.
    # Actually, let's just use it.
    
    best_points = w2_array
    best_k = len(best_points)
    
    # Strategy 2: Weight 4 vectors (Norm^2 = 4)
    # Condition: dist^2 >= 4 => 8 - 2*u.v >= 4 => u.v <= 2
    # Pool size: 5280
    if time.time() < deadline - 60:
        w4_vectors = generate_weight_vectors(d, 4)
        # Use greedy selection
        # Time limit for this part: 30 seconds
        selected_w4 = greedy_select(w4_vectors, max_dot_product=2, time_limit=30)
        if len(selected_w4) > best_k:
            best_points = selected_w4
            best_k = len(best_points)
            
    # Strategy 3: Weight 3 vectors (Norm^2 = 3)
    # Condition: dist^2 >= 3 => 6 - 2*u.v >= 3 => 2*u.v <= 3 => u.v <= 1.5 => u.v <= 1
    # Pool size: 1320
    if time.time() < deadline - 30:
        w3_vectors = generate_weight_vectors(d, 3)
        selected_w3 = greedy_select(w3_vectors, max_dot_product=1, time_limit=15)
        if len(selected_w3) > best_k:
            best_points = selected_w3
            best_k = len(best_points)

    return best_points
