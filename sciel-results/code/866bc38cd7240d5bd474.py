import numpy as np
import itertools
import time
import random
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 610  # safety margin before 630s limit

    # Strategy: Model as Maximum Independent Set on a graph of all 11,858 integer
    # vectors with squared norm 8 (coordinates in {-2, 0, 2}).
    # An edge exists if the squared distance between two vectors is less than 8.
    # Solve using a fast greedy coloring-based branch-and-bound algorithm.

    # Generate all candidate vectors with coordinates in {-2, 0, 2} and norm_sq == 8
    # This means exactly two non-zero entries, each being ±2.
    # Number of such vectors: C(11, 2) * 4 = 55 * 4 = 220.
    # Wait, that's only 220. Let me reconsider.
    # 
    # Actually, norm_sq == 8 with entries in {-2, 0, 2}:
    # - Two entries of ±2: norm_sq = 4 + 4 = 8. Count: C(11,2) * 4 = 55 * 4 = 220
    # - One entry of ±2 and two entries of ±? No, entries are only {-2, 0, 2}.
    # So only 220 candidates. That's not 11,858.
    #
    # Let me reconsider the problem. The strategy mentions 11,858 vectors with norm_sq 8.
    # Perhaps it means entries in {-1, 0, 1, 2, -2} or similar?
    # Or perhaps norm_sq <= 8?
    #
    # Let me try: all integer vectors with entries in {-2, -1, 0, 1, 2} and norm_sq == 8.
    # This is more complex. Let me just use the 220 vectors with exactly two ±2 entries.
    # Actually, let me use a broader pool: all vectors with norm_sq <= 8 and entries in {-2,-1,0,1,2}.
    
    # For now, let's use the 220 vectors with exactly two non-zero entries each ±2.
    # These all have norm_sq = 8.
    # The constraint is: min pairwise dist_sq >= max norm_sq = 8.
    # So we need pairwise dist_sq >= 8.
    
    candidates = []
    cand_set = set()
    
    # Type: exactly two entries of ±2
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = 2 * si
                    p[j] = 2 * sj
                    t = tuple(p)
                    if t not in cand_set:
                        cand_set.add(t)
                        candidates.append(p)
    
    n_cand = len(candidates)
    print(f"Number of candidates (norm_sq == 8, two ±2 entries): {n_cand}")
    
    # Build conflict graph: edge (i,j) if dist_sq(i,j) < 8
    cand_arr = np.array(candidates, dtype=np.int64)
    norms_sq = np.array([np.dot(p, p) for p in candidates], dtype=np.int64)
    
    # All norms are 8, so dist_sq = 8 + 8 - 2*dot = 16 - 2*dot
    # Conflict if dist_sq < 8, i.e., 16 - 2*dot < 8, i.e., dot > 4.
    # Since entries are in {-2, 0, 2}, dot product is a multiple of 4.
    # dot > 4 means dot >= 8. But max dot for two norm-8 vectors is 8 (when they're the same).
    # So conflict if dot >= 8, which means they're the same vector (dot = 8).
    # Wait, that means no two distinct vectors conflict? That can't be right.
    # 
    # Let me recalculate. If p and q both have norm_sq = 8, then:
    # dist_sq = ||p - q||^2 = ||p||^2 + ||q||^2 - 2*p·q = 16 - 2*p·q
    # Conflict if dist_sq < 8, so 16 - 2*p·q < 8, meaning p·q > 4.
    # 
    # For two vectors with entries in {-2, 0, 2} and exactly two non-zero entries:
    # p·q can be: 
    # - 8 if p == q
    # - 4 if they share one coordinate with same sign (e.g., p=(2,2,0...), q=(2,-2,0...), dot=4-4=0? No)
    # Let me think more carefully.
    # p = (2, 2, 0, ..., 0), q = (2, -2, 0, ..., 0): dot = 4 - 4 = 0
    # p = (2, 2, 0, ..., 0), q = (2, 2, 0, ..., 0): dot = 8
    # p = (2, 2, 0, ..., 0), q = (-2, 2, 0, ..., 0): dot = -4 + 4 = 0
    # p = (2, 2, 0, ..., 0), q = (2, 0, 2, 0, ...): dot = 4 + 0 = 4
    # p = (2, 2, 0, ..., 0), q = (0, 2, 2, 0, ...): dot = 0 + 4 = 4
    # 
    # So p·q can be -8, -4, 0, 4, 8.
    # Conflict if p·q > 4, so only if p·q = 8, which means p == q.
    # So no two distinct vectors conflict! We can take all 220.
    
    # But wait, the max norm_sq is 8, and min dist_sq must be >= 8.
    # If we take all 220, the min dist_sq is the minimum over all pairs.
    # For p=(2,2,0...), q=(2,0,2,0...): dist_sq = 16 - 2*4 = 8. OK.
    # For p=(2,2,0...), q=(-2,2,0...): dist_sq = 16 - 2*0 = 16. OK.
    # So all pairs have dist_sq >= 8. We can take all 220!
    
    # But that's only 220. The current score is 219, so this is an improvement.
    # But can we do better by including other vectors?
    
    # Let me also include vectors with norm_sq < 8, as long as the overall constraint holds.
    # If we include a vector with norm_sq < 8, the max norm_sq might still be 8.
    # Then all pairwise distances must be >= 8.
    
    # Let me try: include all vectors with norm_sq <= 8 and entries in {-2,-1,0,1,2}.
    # This is a larger pool.
    
    # For now, let's just return the 220 vectors and see if that's valid.
    points = candidates[:]
    points_array = np.array(points, dtype=np.int64)
    
    # Verify
    if len(points_array) >= 2:
        max_norm_sq = np.max(np.sum(points_array ** 2, axis=1))
        min_dist_sq = float('inf')
        for i in range(len(points_array)):
            for j in range(i + 1, len(points_array)):
                diff = points_array[i] - points_array[j]
                dsq = np.dot(diff, diff)
                if dsq < min_dist_sq:
                    min_dist_sq = dsq
        print(f"Max norm_sq: {max_norm_sq}, Min dist_sq: {min_dist_sq}")
        if min_dist_sq < max_norm_sq:
            # Invalid, fallback
            points = []
            for i in range(d):
                p1 = np.zeros(d, dtype=np.int64)
                p1[i] = 2
                points.append(p1)
                p2 = np.zeros(d, dtype=np.int64)
                p2[i] = -2
                points.append(p2)
            points_array = np.array(points, dtype=np.int64)
    
    return points_array
