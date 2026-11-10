import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0
    
    d = 11
    
    # Generate all weight-2 vectors (norm^2 = 2)
    w2_vectors = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * d
                    v[i] = s1
                    v[j] = s2
                    w2_vectors.append(v)
    
    # All 220 weight-2 vectors are mutually compatible
    # For any two distinct w2 vectors u, v: u.v <= 1, so ||u-v||^2 = 4 - 2*u.v >= 2 = max_norm^2
    selected = list(w2_vectors)
    
    # Try to add weight-4 vectors (norm^2 = 4)
    # Constraint with w2: w2.v <= 1 (since min(2,4)/2 = 1)
    # Constraint with other w4: w4a.w4b <= 2 (since min(4,4)/2 = 2)
    
    w4_vectors = []
    for indices in combinations(range(d), 4):
        for signs in product([1, -1], repeat=4):
            v = [0] * d
            for idx, s in zip(indices, signs):
                v[idx] = s
            w4_vectors.append(v)
    
    if time.time() > deadline - 30:
        return np.array(selected, dtype=np.int64)
    
    # Precompute w2 as array for fast dot products
    w2_arr = np.array(w2_vectors, dtype=np.int64)
    w4_arr = np.array(w4_vectors, dtype=np.int64)
    
    # Check w4 against all w2: need all dots <= 1
    w4_w2_dots = w4_arr @ w2_arr.T  # shape (len(w4), len(w2))
    w4_valid_w2 = np.all(w4_w2_dots <= 1, axis=1)
    
    candidate_indices = np.where(w4_valid_w2)[0]
    candidate_w4 = [w4_vectors[i] for i in candidate_indices]
    
    if time.time() > deadline - 30:
        return np.array(selected, dtype=np.int64)
    
    # Greedily add w4 vectors
    selected_w4 = []
    selected_w4_arr = []
    
    for v in candidate_w4:
        if time.time() > deadline - 20:
            break
        
        v_arr = np.array(v, dtype=np.int64)
        
        valid = True
        if selected_w4_arr:
            dots = selected_w4_arr @ v_arr
            if np.any(dots > 2):
                valid = False
        
        if valid:
            selected_w4.append(v)
            if not selected_w4_arr:
                selected_w4_arr = v_arr.reshape(1, -1)
            else:
                selected_w4_arr = np.vstack([selected_w4_arr, v_arr.reshape(1, -1)])
    
    all_points = selected + selected_w4
    
    if len(all_points) < 2:
        return np.array(selected, dtype=np.int64)
    
    points = np.array(all_points, dtype=np.int64)
    
    # Final verification
    k = points.shape[0]
    norms2 = np.sum(points**2, axis=1)
    max_norm2 = int(np.max(norms2))
    
    min_dist2 = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            d2 = int(np.dot(diff, diff))
            if d2 < min_dist2:
                min_dist2 = d2
        if time.time() > deadline - 10:
            break
    
    if np.any(np.all(points == 0, axis=1)):
        return np.array(selected, dtype=np.int64)
    
    if min_dist2 < max_norm2:
        points = np.array(selected, dtype=np.int64)
    
    points = np.around(points).astype(np.int64)
    return points
