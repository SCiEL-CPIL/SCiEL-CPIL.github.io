import numpy as np
import itertools
import time
import random

def generate_candidates(max_norm_sq=4, d=11):
    """
    Generate all integer vectors in Z^d with squared norm <= max_norm_sq.
    """
    candidates = []
    # We need to generate vectors efficiently.
    # For norm 2: 2 non-zero entries (+/- 1).
    # For norm 3: 3 non-zero entries (+/- 1).
    # For norm 4: 
    #   4 non-zero entries (+/- 1)
    #   1 non-zero entry (+/- 2)
    
    # To keep it simple and fast, we'll generate based on support size and values.
    
    # Norm 2
    for idxs in itertools.combinations(range(d), 2):
        i, j = idxs
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Norm 3
    for idxs in itertools.combinations(range(d), 3):
        i, j, k = idxs
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
                    
    # Norm 4
    # Case 1: Four +/- 1s
    for idxs in itertools.combinations(range(d), 4):
        i, j, k, l = idxs
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    for sl in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates.append(v)
                        
    # Case 2: One +/- 2
    for i in range(d):
        for si in [2, -2]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = si
            candidates.append(v)
            
    return np.array(candidates)

def is_valid_subset(points, max_norm_sq):
    """
    Check if the set of points is valid.
    Condition: min_dist_sq >= max_norm_sq_of_points
    """
    if len(points) < 2:
        return True
    norms_sq = np.sum(points**2, axis=1)
    max_norm_sq_actual = np.max(norms_sq)
    
    min_dist_sq = float('inf')
    # Optimization: If we know the target min_dist_sq, we can break early.
    # But here we just check validity.
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            diff = points[i] - points[j]
            d_sq = np.sum(diff**2)
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                if min_dist_sq < max_norm_sq_actual:
                    return False
    return min_dist_sq >= max_norm_sq_actual

def greedy_pack(candidates, d=11, time_limit=500):
    """
    Greedy packing with local search.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Sort candidates by norm (ascending) to pack small ones first?
    # Or random order?
    # Let's try random order with multiple restarts.
    
    best_points = np.array([]).reshape(0, d)
    best_size = 0
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle candidates
        indices = np.arange(len(candidates))
        np.random.shuffle(indices)
        shuffled_cands = candidates[indices]
        
        current_points = []
        
        for v in shuffled_cands:
            if time.time() > deadline:
                break
            
            # Try adding v
            test_points = np.vstack([np.array(current_points), v]) if current_points else v.reshape(1, -1)
            
            # Quick check: max norm
            norms_sq = np.sum(test_points**2, axis=1)
            max_norm_sq = np.max(norms_sq)
            
            # Check distances
            valid = True
            # Only need to check distance of new point v against existing points
            # and update min_dist_sq
            if current_points:
                existing = np.array(current_points)
                diffs = existing - v
                dists_sq = np.sum(diffs**2, axis=1)
                min_dist_to_new = np.min(dists_sq)
                
                # We also need the global min_dist_sq of the existing set
                # To avoid recalculating all pairs, we can maintain it, 
                # but for simplicity in this greedy step, we check the new one.
                # However, adding a point doesn't change distances between existing points.
                # So global min_dist_sq = min(old_min_dist_sq, min_dist_to_new).
                
                # We don't store old_min_dist_sq in this simple loop.
                # So we just check if min_dist_to_new >= max_norm_sq.
                # This is a necessary condition, but not sufficient for the global min_dist_sq
                # if the global min_dist_sq was smaller than max_norm_sq?
                # No, if the set was valid before, old_min_dist_sq >= old_max_norm_sq.
                # new_max_norm_sq = max(old_max_norm_sq, norm(v)).
                # We need new_min_dist_sq >= new_max_norm_sq.
                # new_min_dist_sq = min(old_min_dist_sq, min_dist_to_new).
                # If old set was valid, old_min_dist_sq >= old_max_norm_sq.
                # If norm(v) <= old_max_norm_sq, then new_max_norm_sq = old_max_norm_sq.
                # We need min(old_min_dist_sq, min_dist_to_new) >= old_max_norm_sq.
                # Since old_min_dist_sq >= old_max_norm_sq, we just need min_dist_to_new >= old_max_norm_sq.
                
                # If norm(v) > old_max_norm_sq, then new_max_norm_sq = norm(v).
                # We need min(old_min_dist_sq, min_dist_to_new) >= norm(v).
                # This implies old_min_dist_sq >= norm(v) AND min_dist_to_new >= norm(v).
                
                # To be safe and simple, let's just do the full check for small sets, 
                # or assume greedy order keeps norms low.
                
                if min_dist_to_new < max_norm_sq:
                    valid = False
                else:
                    # We must also ensure that the existing set's min_dist_sq is >= max_norm_sq.
                    # Since we built it greedily, it should be.
                    # But if we added a high norm vector, it might break the condition for existing pairs?
                    # No, existing pairs distance is fixed.
                    # But max_norm_sq increased.
                    # So we need to check if old_min_dist_sq >= new_max_norm_sq.
                    # We didn't track old_min_dist_sq.
                    # So this greedy check is flawed for mixed norms.
                    
                    # Let's refine: Keep track of current_min_dist_sq.
                    pass
            
            # Simplified approach for correctness:
            # Just use the full validation for the candidate set if size is small, 
            # or maintain state.
            
            # Let's restart with a stateful greedy.
            break # Break inner loop to use the stateful version below
            
        # Stateful Greedy
        current_points = []
        current_min_dist_sq = float('inf')
        current_max_norm_sq = 0
        
        for v in shuffled_cands:
            if time.time() > deadline:
                break
            
            v_norm_sq = np.sum(v**2)
            new_max_norm_sq = max(current_max_norm_sq, v_norm_sq)
            
            if current_points:
                existing = np.array(current_points)
                diffs = existing - v
                dists_sq = np.sum(diffs**2, axis=1)
                min_dist_to_new = np.min(dists_sq)
                
                new_min_dist_sq = min(current_min_dist_sq, min_dist_to_new)
            else:
                new_min_dist_sq = float('inf')
                new_max_norm_sq = v_norm_sq # Only 1 point, dist is inf? 
                # Condition: min_dist >= max_norm. 
                # If k=1, min_dist is undefined or inf. 
                # Task says k>=2. 
                # For k=1, we can't return. 
                # So we only care when k>=2.
                
            # Check validity condition: new_min_dist_sq >= new_max_norm_sq
            # Note: If current_points is empty, new_min_dist_sq is inf.
            # If len(current_points) == 1, new_min_dist_sq is dist(p0, v).
            
            if new_min_dist_sq >= new_max_norm_sq:
                current_points.append(v)
                current_min_dist_sq = new_min_dist_sq
                current_max_norm_sq = new_max_norm_sq
                
        if len(current_points) > best_size:
            best_size = len(current_points)
            best_points = np.array(current_points)
            
    return best_points

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    # Fallback: The known 220 solution
    def generate_norm2_vectors(d=11):
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
    
    fallback_points = generate_norm2_vectors(11)
    fallback_points = np.around(fallback_points).astype(np.int64)
    
    # Try to find a better solution using greedy packing on a larger candidate set
    # We limit the candidate set to norm 2 and 3 to keep it manageable
    # Generating norm 4 might be too slow or yield worse results due to sparsity
    
    try:
        # Generate candidates for norm 2 and 3
        # Norm 2: 220 vectors
        # Norm 3: 1320 vectors
        # Total: 1540 vectors.
        
        # We will run the greedy search.
        # To save time, we pre-generate.
        candidates = generate_candidates(max_norm_sq=3, d=11)
        
        # Run greedy pack
        best_points = greedy_pack(candidates, d=11, time_limit=500)
        
        if len(best_points) > len(fallback_points):
            # Validate the best points found
            if is_valid_subset(best_points, np.max(np.sum(best_points**2, axis=1))):
                best_points = np.around(best_points).astype(np.int64)
                return best_points
        else:
            # If greedy didn't beat 220, return 220
            return fallback_points
            
    except Exception as e:
        # If anything goes wrong, return fallback
        return fallback_points

    return fallback_points
