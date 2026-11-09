# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points such that
    minimum squared pairwise distance >= maximum squared norm.
    Returns np.ndarray of shape (k, 11) with int64 entries.
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s kill

    d = 11

    # Base construction: all vectors with exactly 2 non-zero entries ±1
    # C(11,2) * 4 = 55 * 4 = 220 points
    points_list = []
    for i, j in combinations(range(d), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)

    points = np.array(points_list, dtype=np.int64)
    k = points.shape[0]

    # Quick validity check
    norms_sq = np.sum(points**2, axis=1)
    if k < 2 or np.any(norms_sq == 0):
        return np.array([[1,0,0,0,0,0,0,0,0,0,0],
                         [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    max_norm_sq = int(np.max(norms_sq))

    # Verify min pairwise dist >= max norm
    valid = True
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            dist_sq = int(np.dot(diff, diff))
            if dist_sq < max_norm_sq:
                valid = False
                break
        if not valid:
            break

    if not valid:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0],
                         [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    best_points = points.copy()
    best_k = k

    # Strategy: try to build a larger set using vectors with entries in {-2,-1,0,1,2}
    # and small Hamming weight, using greedy addition.
    
    # Generate candidate pool: all vectors with entries in {-2,-1,0,1,2}
    # with Hamming weight 1 or 2 (i.e., at most 2 non-zero entries)
    # Weight 1: 11 * 4 = 44 vectors (±1, ±2)
    # Weight 2: C(11,2) * 4 * 4 = 55 * 16 = 880 vectors
    # Total: 924 candidates
    
    candidates = []
    
    # Weight 1 vectors
    for i in range(d):
        for val in (-2, -1, 1, 2):
            p = np.zeros(d, dtype=np.int64)
            p[i] = val
            candidates.append(p)
    
    # Weight 2 vectors
    for i, j in combinations(range(d), 2):
        for vi in (-2, -1, 1, 2):
            for vj in (-2, -1, 1, 2):
                p = np.zeros(d, dtype=np.int64)
                p[i] = vi
                p[j] = vj
                candidates.append(p)
    
    candidate_arr = np.array(candidates, dtype=np.int64)
    n_cand = len(candidate_arr)
    
    # Greedy: start with the 220 weight-2 ±1 vectors, try adding others
    # Current set has max_norm_sq = 2
    # We can add vectors with norm_sq up to 2 initially
    # If we add a vector with norm_sq > 2, max_norm increases and all pairs must have dist >= new max
    
    # Let's try a different approach: build from scratch using greedy
    # Start with empty set, add vectors in order of increasing norm, then by some heuristic
    
    # Sort candidates by norm squared, then by lex order
    cand_norms = np.array([int(np.dot(c, c)) for c in candidates])
    order = np.argsort(cand_norms, kind='stable')
    sorted_candidates = candidate_arr[order]
    sorted_norms = cand_norms[order]
    
    # Greedy construction
    current_set = []
    current_max_norm = 0
    
    for idx in range(n_cand):
        if time.time() > deadline - 30:
            break
        
        cand = sorted_candidates[idx]
        cnorm = int(sorted_norms[idx])
        
        if cnorm == 0:
            continue
        
        new_max_norm = max(current_max_norm, cnorm)
        
        # Check if cand can be added to current_set
        ok = True
        for existing in current_set:
            diff = cand - existing
            dsq = int(np.dot(diff, diff))
            if dsq < new_max_norm:
                ok = False
                break
        
        if ok:
            current_set.append(cand)
            current_max_norm = new_max_norm
    
    current_arr = np.array(current_set, dtype=np.int64) if current_set else np.zeros((0, d), dtype=np.int64)
    current_k = len(current_set)
    
    if current_k > best_k:
        best_k = current_k
        best_points = current_arr.copy()
    
    # Also try the original 220 + weight-1 vectors approach
    # which we know gives 220 + 22 = 242
    if time.time() < deadline - 60:
        set2 = list(points_list)  # 220 weight-2 ±1
        for i in range(d):
            for s in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = s
                # Check if can be added
                ok = True
                for existing in set2:
                    diff = p - existing
                    dsq = int(np.dot(diff, diff))
                    if dsq < 2:
                        ok = False
                        break
                if ok:
                    set2.append(p)
        
        arr2 = np.array(set2, dtype=np.int64)
        if len(set2) > best_k:
            best_k = len(set2)
            best_points = arr2.copy()
    
    # Try adding weight-2 vectors with values ±2 to the 220 set
    # A vector like (2, 1) has norm_sq = 5. If added, max_norm becomes 5.
    # dist((2,1), (1,1)) = (1,0) -> dist_sq = 1 < 5. Not ok.
    # dist((2,1), (1,-1)) = (1,2) -> dist_sq = 5 >= 5. Ok.
    # But dist((2,1), (-1,1)) = (3,0) -> dist_sq = 9 >= 5. Ok.
    # dist((2,1), (-1,-1)) = (3,2) -> dist_sq = 13 >= 5. Ok.
    # So (2,1) can coexist with (1,-1), (-1,1), (-1,-1) but not (1,1).
    # This gets complex. The greedy above should handle it.
    
    # Try a local search / hill climbing on the greedy result
    # Remove one point, try to add multiple new points
    if time.time() < deadline - 30 and best_k > 0:
        # Try removing each point from best_points and see if we can add more
        bp = best_points
        bk = best_k
        
        # For efficiency, only try if we have time
        # Try a few random removals
        np.random.seed(42)
        for trial in range(5):
            if time.time() > deadline - 30:
                break
            
            # Pick a random point to remove
            remove_idx = np.random.randint(0, bk)
            modified_set = list(bp)
            modified_set.pop(remove_idx)
            modified_arr = np.array(modified_set, dtype=np.int64)
            modified_max_norm = int(np.max(np.sum(modified_arr**2, axis=1)))
            
            # Try to add candidates
            improved = False
            for idx in range(n_cand):
                if time.time() > deadline - 30:
                    break
                
                cand = sorted_candidates[idx]
                cnorm = int(sorted_norms[idx])
                
                if cnorm == 0:
                    continue
                
                # Check if already in set
                already = False
                for existing in modified_set:
                    if np.array_equal(cand, existing):
                        already = True
                        break
                if already:
                    continue
                
                new_max_norm = max(modified_max_norm, cnorm)
                
                ok = True
                for existing in modified_set:
                    diff = cand - existing
                    dsq = int(np.dot(diff, diff))
                    if dsq < new_max_norm:
                        ok = False
                        break
                
                if ok:
                    modified_set.append(cand)
                    modified_max_norm = new_max_norm
                    improved = True
            
            if len(modified_set) > best_k:
                best_k = len(modified_set)
                best_points = np.array(modified_set, dtype=np.int64)
    
    # Final validity check
    if best_k >= 2:
        norms_sq_final = np.sum(best_points**2, axis=1)
        max_norm_final = int(np.max(norms_sq_final))
        
        valid_final = True
        for i in range(best_k):
            for j in range(i + 1, best_k):
                diff = best_points[i] - best_points[j]
                dist_sq = int(np.dot(diff, diff))
                if dist_sq < max_norm_final:
                    valid_final = False
                    break
            if not valid_final:
                break
        
        if not valid_final:
            # Fall back to the 220 weight-2 vectors
            return np.around(points).astype(np.int64)
    
    result = np.around(best_points).astype(np.int64)
    return result


# EVOLVE-BLOCK-END
