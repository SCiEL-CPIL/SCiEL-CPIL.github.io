import numpy as np
import itertools
import time
import random

def generate_norm2_vectors(d=11):
    """
    Generate all integer vectors in Z^d with squared norm 2.
    These are vectors with exactly two non-zero entries, each +/- 1.
    """
    vecs = []
    for idxs in itertools.combinations(range(d), 2):
        i, j = idxs
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                vecs.append(v)
    return np.array(vecs)

def validate_configuration(points):
    """
    Check if the set of points is valid.
    Condition: min_dist_sq >= max_norm_sq_of_points, and k >= 2, no zero vector.
    """
    if points.shape[1] != 11:
        return False
    k = len(points)
    if k < 2:
        return False
    # Check no zero vector
    for p in points:
        if np.all(p == 0):
            return False
    norms_sq = np.sum(points**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    if max_norm_sq == 0:
        return False
    # Check all pairwise distances
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            d_sq = int(np.sum(diff**2))
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
            if min_dist_sq < max_norm_sq:
                return False
    return min_dist_sq >= max_norm_sq

def greedy_pack_with_state(candidates, d=11, deadline=None):
    """
    Greedy packing with state tracking.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    best_points = np.array([]).reshape(0, d)
    best_size = 0
    best_min_dist = float('inf')
    best_max_norm = 0
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        indices = np.arange(len(candidates))
        np.random.shuffle(indices)
        shuffled_cands = candidates[indices]
        
        current_points = []
        current_min_dist_sq = float('inf')
        current_max_norm_sq = 0
        
        for v in shuffled_cands:
            if time.time() > deadline:
                break
            
            v_norm_sq = int(np.sum(v**2))
            new_max_norm_sq = max(current_max_norm_sq, v_norm_sq)
            
            if current_points:
                existing = np.array(current_points)
                diffs = existing - v
                dists_sq = np.sum(diffs**2, axis=1)
                min_dist_to_new = int(np.min(dists_sq))
                new_min_dist_sq = min(current_min_dist_sq, min_dist_to_new)
            else:
                new_min_dist_sq = float('inf')
            
            if new_min_dist_sq >= new_max_norm_sq:
                current_points.append(v)
                current_min_dist_sq = new_min_dist_sq
                current_max_norm_sq = new_max_norm_sq
        
        if len(current_points) > best_size:
            best_size = len(current_points)
            best_points = np.array(current_points)
            best_min_dist = current_min_dist_sq
            best_max_norm = current_max_norm_sq
    
    return best_points

def maximum_clique_greedy(adj_matrix):
    """
    Find a large clique using greedy heuristic.
    adj_matrix[i][j] = 1 if i and j can coexist (distance >= threshold).
    We want maximum set where all pairs are compatible.
    """
    n = adj_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    clique = []
    
    # Order vertices by degree (descending)
    degrees = np.sum(adj_matrix, axis=1)
    order = np.argsort(-degrees)
    
    for v in order:
        if visited[v]:
            continue
        # Check if v is compatible with all in clique
        compatible = True
        for u in clique:
            if not adj_matrix[v, u]:
                compatible = False
                break
        if compatible:
            clique.append(v)
            visited[v] = True
    
    return clique

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    # Generate the 220 norm-2 vectors
    points_220 = generate_norm2_vectors(11)
    points_220 = np.around(points_220).astype(np.int64)
    
    # These 220 vectors form a valid configuration:
    # min_dist_sq between any two = 2 (when they share a coordinate with same sign, distance is 2)
    # Actually, let's verify: 
    # Two norm-2 vectors u, v. 
    # If they are orthogonal (no shared non-zero coords), dist^2 = 4.
    # If they share one coord with same sign, e.g., (1,1,0...) and (1,-1,0...), dist^2 = 0 + 4 = 4.
    # If they share one coord with opposite sign? No, they have same sign if same coordinate index.
    # Let's just validate.
    
    if validate_configuration(points_220):
        # Try to improve with greedy search including norm 3 vectors
        try:
            # Generate norm 3 vectors
            candidates = []
            # Norm 2
            for idxs in itertools.combinations(range(11), 2):
                i, j = idxs
                for si in [1, -1]:
                    for sj in [1, -1]:
                        v = np.zeros(11, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        candidates.append(v)
            
            # Norm 3
            for idxs in itertools.combinations(range(11), 3):
                i, j, k = idxs
                for si in [1, -1]:
                    for sj in [1, -1]:
                        for sk in [1, -1]:
                            v = np.zeros(11, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
            
            candidates = np.array(candidates)
            
            # Run greedy packing
            best_points = greedy_pack_with_state(candidates, d=11, deadline=deadline)
            
            if len(best_points) > 220 and validate_configuration(best_points):
                best_points = np.around(best_points).astype(np.int64)
                return best_points
        
        except Exception:
            pass
    
    # Return the 220 solution
    return points_220
