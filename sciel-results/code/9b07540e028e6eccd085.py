# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580
    
    def generate_vectors(n_nonzero):
        """Generate all vectors with exactly n_nonzero non-zero entries (each ±1)."""
        vecs = []
        for combo in itertools.combinations(range(d), n_nonzero):
            for signs in itertools.product([1, -1], repeat=n_nonzero):
                v = np.zeros(d, dtype=np.int64)
                for idx, val in zip(combo, signs):
                    v[idx] = val
                vecs.append(v)
        return vecs
    
    # Generate candidates by norm level
    candidates_2 = generate_vectors(2)
    candidates_4 = generate_vectors(4)
    candidates_6 = generate_vectors(6)
    candidates_8 = generate_vectors(8)
    candidates_10 = generate_vectors(10)
    
    # Pre-compute norms
    norm2_2 = 2
    norm2_4 = 4
    norm2_6 = 6
    norm2_8 = 8
    norm2_10 = 10
    
    def greedy_select(candidate_list, norm_sq, existing_points, existing_norms, max_norm):
        """
        Greedily select from candidate_list.
        existing_points: list of selected np arrays
        existing_norms: list of squared norms of selected points
        max_norm: the global maximum squared norm (determines distance constraint)
        Returns list of newly selected vectors.
        """
        selected = []
        sel_arr = np.zeros((0, d), dtype=np.int64)
        
        for v in candidate_list:
            if time.time() > deadline - 5:
                break
            
            # Check against all existing points
            valid = True
            
            # Check against existing selected points
            if len(existing_points) > 0:
                ex_arr = np.array(existing_points, dtype=np.int64)
                dots = ex_arr @ v
                # Need ||u-v||^2 >= max_norm for all existing u
                # ||u-v||^2 = norm_u + norm_sq - 2*dot >= max_norm
                # dot <= (norm_u + norm_sq - max_norm) / 2
                for i, dot in enumerate(dots):
                    allowed = (existing_norms[i] + norm_sq - max_norm) / 2.0
                    if dot > allowed:
                        valid = False
                        break
            
            # Check against newly selected in this call
            if valid and len(selected) > 0:
                if sel_arr.shape[0] == 0:
                    sel_arr = np.array(selected, dtype=np.int64)
                dots = sel_arr @ v
                for dot in dots:
                    allowed = (norm_sq + norm_sq - max_norm) / 2.0
                    if dot > allowed:
                        valid = False
                        break
            
            if valid:
                selected.append(v)
                sel_arr = np.vstack((sel_arr, v.reshape(1, -1).astype(np.int64)))
        
        return selected
    
    def validate_and_count(points):
        """Validate the set and return count if valid, else 0."""
        if len(points) < 2:
            return 0
        arr = np.array(points, dtype=np.int64)
        norms = np.sum(arr * arr, axis=1)
        max_norm = int(np.max(norms))
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                diff = arr[i] - arr[j]
                sq_dist = int(np.sum(diff * diff))
                if sq_dist < max_norm:
                    return 0
        return n
    
    best_points = []
    
    # Strategy 1: Try different max_norm levels and greedy orders
    # For each possible max_norm, try to pack as many points as possible
    # max_norm must be even (since all norms are even for ±1 vectors)
    # Possible max_norm values: 2, 4, 6, 8, 10
    
    strategies = [
        # (max_norm, [norm_levels_to_use in order])
        (4, [4, 6, 2]),
        (4, [4, 2, 6]),
        (4, [6, 4, 2]),
        (4, [2, 4, 6]),
        (6, [4, 6, 2]),
        (6, [6, 4, 2]),
        (6, [4, 2, 6]),
        (6, [2, 4, 6]),
        (6, [6, 2, 4]),
        (8, [4, 6, 8, 2]),
        (8, [6, 4, 8, 2]),
        (8, [8, 4, 6, 2]),
        (10, [4, 6, 8, 10, 2]),
        (10, [6, 4, 8, 10, 2]),
        (2, [2, 4, 6]),
        (2, [4, 2, 6]),
    ]
    
    # Map norm level to candidates
    norm_to_candidates = {
        2: (candidates_2, norm2_2),
        4: (candidates_4, norm2_4),
        6: (candidates_6, norm2_6),
        8: (candidates_8, norm2_8),
        10: (candidates_10, norm2_10),
    }
    
    for max_norm, norm_order in strategies:
        if time.time() > deadline - 30:
            break
        
        # Only use norm levels <= max_norm
        valid_order = [n for n in norm_order if n <= max_norm]
        if not valid_order:
            continue
        
        existing_points = []
        existing_norms = []
        
        for norm_level in valid_order:
            if time.time() > deadline - 10:
                break
            cands, nsq = norm_to_candidates[norm_level]
            new_sel = greedy_select(cands, nsq, existing_points, existing_norms, max_norm)
            existing_points.extend(new_sel)
            existing_norms.extend([nsq] * len(new_sel))
        
        # Validate
        cnt = validate_and_count(existing_points)
        if cnt > len(best_points):
            best_points = list(existing_points)
    
    # Strategy 2: Randomized greedy with multiple restarts
    rng = random.Random(42)
    for restart in range(20):
        if time.time() > deadline - 15:
            break
        
        # Pick a random max_norm from {4, 6, 8}
        max_norm = rng.choice([4, 6, 8])
        
        # Randomly order the norm levels
        norm_levels = [n for n in [2, 4, 6, 8, 10] if n <= max_norm]
        rng.shuffle(norm_levels)
        
        existing_points = []
        existing_norms = []
        
        for norm_level in norm_levels:
            if time.time() > deadline - 5:
                break
            cands, nsq = norm_to_candidates[norm_level]
            # Shuffle candidates
            shuffled_cands = list(cands)
            rng.shuffle(shuffled_cands)
            new_sel = greedy_select(shuffled_cands, nsq, existing_points, existing_norms, max_norm)
            existing_points.extend(new_sel)
            existing_norms.extend([nsq] * len(new_sel))
        
        cnt = validate_and_count(existing_points)
        if cnt > len(best_points):
            best_points = list(existing_points)
    
    # Fallback if nothing found
    if not best_points:
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                p1 = np.zeros(d, dtype=np.int64); p1[i] = 1; p1[j] = 1; points.append(p1)
                p2 = np.zeros(d, dtype=np.int64); p2[i] = -1; p2[j] = -1; points.append(p2)
        return np.array(points, dtype=np.int64)
    
    result = np.array(best_points, dtype=np.int64)
    return np.around(result).astype(np.int64)

# EVOLVE-BLOCK-END
