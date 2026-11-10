import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.

    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    d = 11
    deadline = time.time() + 580  # Safety margin
    
    # Generate candidate vectors
    # We consider vectors with small integer coordinates.
    # Norm squared <= 4 covers:
    # 1. +-e_i (22 vectors)
    # 2. +-e_i +-e_j (110 vectors)
    # 3. 2*e_i (11 vectors)
    # 4. Combinations of 4 +-e_i (5280 vectors)
    
    candidates = []
    
    # Type 1: Norm 1
    for i in range(d):
        v1 = np.zeros(d, dtype=np.int64)
        v1[i] = 1
        candidates.append(v1)
        v2 = -v1
        candidates.append(v2)
        
    # Type 2: Norm 2
    for i in range(d):
        for j in range(i+1, d):
            # e_i + e_j
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1; v[j] = 1
            candidates.append(v)
            candidates.append(-v)
            
            # e_i - e_j
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1; v[j] = -1
            candidates.append(v)
            candidates.append(-v)
            
    # Type 3: Norm 4 (2*e_i)
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 2
        candidates.append(v)
        candidates.append(-v)
        
    # Type 4: Norm 4 (4 distinct indices)
    # This is a large number, 5280.
    # We might not need all of them for the greedy to find a good set,
    # but let's include them if time permits.
    # Actually, 5280 * 11 is small.
    
    if time.time() < deadline - 10:
        for indices in itertools.combinations(range(d), 4):
            # Generate all 16 sign patterns
            for signs in itertools.product([-1, 1], repeat=4):
                v = np.zeros(d, dtype=np.int64)
                for idx, sign in zip(indices, signs):
                    v[idx] = sign
                candidates.append(v)
                
    candidates = np.array(candidates, dtype=np.int64)
    
    # Remove duplicates just in case (though generation should be unique)
    # And remove zero vector
    candidates = candidates[~np.all(candidates == 0, axis=1)]
    
    # Greedy Selection
    # We want to maximize k.
    # Strategy: Try to find a large set.
    # Since the condition depends on max norm, using smaller norms is generally better for allowing more points.
    # However, the "density" of valid points might be higher for larger norms?
    # Let's try a few random permutations of the candidate list.
    
    best_set = []
    best_k = 0
    
    # Pre-compute norms
    norms = np.sum(candidates * candidates, axis=1)
    
    # Sort by norm ascending? Or random?
    # If we sort by norm, we tend to pick low norm points first.
    # Low norm points have strict distance requirements relative to their norm?
    # Condition: d^2 > N_max.
    # If N_max is small, d^2 must be small but > N_max.
    
    # Let's try 50 random restarts with random shuffle.
    # And one sorted by norm.
    
    num_restarts = 100
    rng = np.random.default_rng(42)
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        if r == 0:
            # Sorted by norm
            order = np.argsort(norms)
        else:
            order = rng.permutation(len(candidates))
            
        selected = []
        max_norm = 0
        min_dist_sq = float('inf')
        
        # For speed, we can check against selected points.
        # k is expected to be small (< 100), so O(k) check per candidate is fine.
        
        for idx in order:
            v = candidates[idx]
            v_norm = norms[idx]
            
            # Potential new max norm
            new_max_norm = max(max_norm, v_norm)
            
            # Check distance to all selected points
            valid = True
            if selected:
                # Compute distances to all selected
                # selected is a list of vectors
                # We can vectorize this
                sel_arr = np.array(selected)
                diffs = sel_arr - v
                dists_sq = np.sum(diffs * diffs, axis=1)
                min_d_to_selected = np.min(dists_sq)
                
                if min_d_to_selected <= new_max_norm:
                    valid = False
            
            if valid:
                # Check if adding this point maintains the global condition
                # The global condition is min_pair_dist_sq > max_norm
                # Current min_pair_dist_sq is min_dist_sq.
                # New min_pair_dist_sq will be min(min_dist_sq, min_d_to_selected)
                # We need new_min > new_max_norm
                
                if selected:
                    new_min_dist = min(min_dist_sq, min_d_to_selected)
                else:
                    new_min_dist = float('inf')
                    
                if new_min_dist > new_max_norm:
                    selected.append(v)
                    max_norm = new_max_norm
                    min_dist_sq = new_min_dist
                    
        k = len(selected)
        if k > best_k:
            best_k = k
            best_set = selected
            
        # Early stopping if we found a very good set?
        # Theoretical max is unknown, but 22 is a baseline.
        # If we find > 50, we can reduce restarts.
        if best_k > 50 and r > 20:
             break

    if not best_set:
        # Fallback to the seed solution
        points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
    else:
        points = np.array(best_set, dtype=np.int64)
        
    # Final check and rounding
    points = np.around(points).astype(np.int64)
    
    return points
