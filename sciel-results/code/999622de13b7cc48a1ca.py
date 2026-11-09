import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    d = 11
    
    # Fallback: D_11 roots (220 points)
    def get_d11_points():
        pts = []
        for i, j in combinations(range(d), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    pts.append(p)
        return np.array(pts)
    
    fallback_pts = get_d11_points()
    best_pts = fallback_pts
    best_k = fallback_pts.shape[0]
    
    # Try to build a larger set from weight-3 vectors
    # Vectors with exactly 3 non-zero entries, each +/- 1
    # Total: C(11, 3) * 2^3 = 165 * 8 = 1320
    
    all_vecs = []
    for indices in combinations(range(d), 3):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[indices[0]] = s1
                    p[indices[1]] = s2
                    p[indices[2]] = s3
                    all_vecs.append(p)
    
    all_vecs = np.array(all_vecs)
    
    # Greedy selection with multiple orderings to find larger independent sets
    # We want max subset where min_dist_sq >= 3
    # Note: Max norm sq is 3. So condition is dist_sq >= 3.
    
    def greedy_select(ordered_vecs, max_selected=None):
        """Greedy selection in the given order."""
        selected = []
        selected_arrays = []
        if max_selected is None:
            max_selected = len(ordered_vecs)
            
        for p in ordered_vecs:
            if len(selected) >= max_selected:
                break
            valid = True
            for q in selected_arrays:
                diff = p - q
                dist_sq = int(np.sum(diff**2))
                if dist_sq < 3:
                    valid = False
                    break
            if valid:
                selected.append(p)
                selected_arrays.append(p)
        return selected
    
    # Try different orderings
    orderings = []
    
    # 1. Original order
    orderings.append(all_vecs.copy())
    
    # 2. Reverse order
    orderings.append(all_vecs[::-1].copy())
    
    # 3. Sort by number of shared indices with previously selected (heuristic)
    # This is harder to precompute, so we'll just try a few random shuffles
    
    # 4. Several random shuffles
    rng = np.random.default_rng(42)
    for trial in range(10):
        shuffled = all_vecs.copy()
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    best_selected = []
    best_selected_len = 0
    
    for ordering in orderings:
        if time.time() > deadline - 30:  # Leave time for verification
            break
        
        selected = greedy_select(ordering)
        
        if len(selected) > best_selected_len:
            best_selected_len = len(selected)
            best_selected = selected
            best_pts = np.array(best_selected)
            best_k = len(best_selected)
    
    # Final verification and return
    if best_pts.shape[1] != d:
        return fallback_pts
    
    # Ensure no zero vectors (should be true for weight 2 and 3)
    # Ensure min dist >= max norm
    
    max_norm_sq = int(np.max(np.sum(best_pts**2, axis=1)))
    min_dist_sq = float('inf')
    
    k = best_pts.shape[0]
    for i in range(k):
        for j in range(i+1, k):
            diff = best_pts[i] - best_pts[j]
            dist_sq = int(np.sum(diff**2))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if min_dist_sq < max_norm_sq:
                # Invalid, return fallback
                return fallback_pts
    
    return best_pts
