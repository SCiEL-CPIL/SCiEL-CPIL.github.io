import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    d = 11
    
    # Generate D_11 roots: 220 points with exactly two non-zero entries (±1)
    # These have norm_sq = 2, and min pairwise dist_sq = 2 (when they share one coordinate with opposite signs)
    # Actually let's verify: v1 = e_i + e_j, v2 = e_i + e_k
    # diff = (0, 1, -1, ...) -> dist_sq = 2. OK.
    # v1 = e_i + e_j, v2 = -e_i - e_j
    # diff = (2, 2, ...) -> dist_sq = 8. OK.
    # v1 = e_i + e_j, v2 = e_i - e_j
    # diff = (0, 2, ...) -> dist_sq = 4. OK.
    # v1 = e_i + e_j, v2 = -e_i + e_j
    # diff = (2, 0, ...) -> dist_sq = 4. OK.
    # So min dist_sq among D_11 roots is 2, which equals max_norm_sq = 2. Valid.
    
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
    
    best_pts = get_d11_points()
    best_k = best_pts.shape[0]  # 220
    
    # Try to find a larger set using weight-3 vectors (±1 in 3 positions)
    # These have norm_sq = 3. If we use ONLY weight-3 vectors, max_norm_sq = 3.
    # We need min dist_sq >= 3.
    # Two weight-3 vectors sharing s indices:
    # - s=0: dist_sq = 6 (OK)
    # - s=1: If signs agree on shared index: dist_sq = 1+4 = 5? Let me compute.
    #   v1 = (1,1,1,0,...), v2 = (1,0,0,1,0,...) share index 0 with same sign.
    #   diff = (0,1,1,-1,0,...) -> dist_sq = 0+1+1+1 = 3. OK (>= 3)
    #   If signs disagree on shared index: v1 = (1,1,1,0,...), v2 = (-1,0,0,1,0,...)
    #   diff = (2,1,1,-1,0,...) -> dist_sq = 4+1+1+1 = 7. OK.
    #   So s=1 is always OK.
    # - s=2: v1 = (1,1,1,0,...), v2 = (1,1,0,1,0,...) share indices 0,1 with same signs.
    #   diff = (0,0,1,-1,0,...) -> dist_sq = 2. NOT OK (< 3).
    #   v1 = (1,1,1,0,...), v2 = (1,-1,0,1,0,...) share index 0 same, index 1 diff.
    #   diff = (0,2,1,-1,0,...) -> dist_sq = 0+4+1+1 = 6. OK.
    #   v1 = (1,1,1,0,...), v2 = (-1,-1,0,1,0,...) share indices 0,1 both diff.
    #   diff = (2,2,1,-1,0,...) -> dist_sq = 4+4+1+1 = 10. OK.
    #   v1 = (1,1,1,0,...), v2 = (-1,1,0,1,0,...) share index 0 diff, index 1 same.
    #   diff = (2,0,1,-1,0,...) -> dist_sq = 4+0+1+1 = 6. OK.
    #   So s=2 is NOT OK only when the two shared indices have the SAME signs.
    #   If they differ in at least one shared index sign, it's OK.
    
    # So the constraint for weight-3 vectors is: no two can share 2 indices with matching signs on both.
    # This is more permissive than just "no two share 2 indices".
    
    # Let's generate all weight-3 vectors and do a greedy selection.
    # Total: C(11,3) * 8 = 165 * 8 = 1320
    
    all_vecs = []
    vec_to_key = {}
    for indices in combinations(range(d), 3):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[indices[0]] = s1
                    p[indices[1]] = s2
                    p[indices[2]] = s3
                    key = (tuple(indices), s1, s2, s3)
                    all_vecs.append(p)
                    vec_to_key[key] = p
    
    all_vecs = np.array(all_vecs)
    
    # For efficient conflict checking, represent each vector as (frozenset of (index, sign) pairs)
    # Two vectors conflict if they share 2 indices AND the signs match on both shared indices.
    
    # Let's build a representation: for each vector, store the set of (index, sign) pairs
    vec_representations = []
    for p in all_vecs:
        reps = set()
        for i in range(d):
            if p[i] != 0:
                reps.add((i, int(p[i])))
        vec_representations.append(frozenset(reps))
    
    # Greedy selection
    selected = []
    selected_reps = []
    
    for i, (p, rep) in enumerate(zip(all_vecs, vec_representations)):
        if time.time() > deadline:
            break
        
        valid = True
        for sel_rep in selected_reps:
            # Check if they share 2 indices with matching signs
            common = rep & sel_rep
            if len(common) >= 2:
                valid = False
                break
        
        if valid:
            selected.append(p)
            selected_reps.append(rep)
    
    w3_k = len(selected)
    
    if w3_k > best_k:
        best_pts = np.array(selected)
        best_k = w3_k
    
    # Verify the result
    if best_pts.shape[1] != d:
        return get_d11_points()
        
    k = best_pts.shape[0]
    if k < 2:
        return get_d11_points()
    
    # Check no zero vector
    if np.any(np.all(best_pts == 0, axis=1)):
        mask = np.any(best_pts != 0, axis=1)
        best_pts = best_pts[mask]
        k = best_pts.shape[0]
    
    # Check min dist sq >= max norm sq
    max_norm_sq = int(np.max(np.sum(best_pts**2, axis=1)))
    
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i+1, k):
            diff = best_pts[i] - best_pts[j]
            dist_sq = int(np.sum(diff**2))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                if min_dist_sq < max_norm_sq:
                    return get_d11_points()
    
    return best_pts
