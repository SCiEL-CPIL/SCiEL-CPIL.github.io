import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    t_start = time.time()
    deadline = t_start + 580  # safety margin before 600s limit

    # Build the 220-point norm^2=2 solution: all vectors with exactly two +-1 entries
    # This is the D11 lattice kissing configuration
    points_220 = _build_220(d)

    # Try to find a larger configuration using norm^2=4 vectors
    # Candidates: vectors with four +-1 entries (5280 total)
    # Conflict condition: dot product > 2 (since norm^2=4, need dist^2 >= 4 iff dot <= 2)
    
    try:
        # Generate type-C vectors: four +-1 entries
        type_c = []
        for combo in combinations(range(d), 4):
            for signs in product([1, -1], repeat=4):
                v = np.zeros(d, dtype=np.int64)
                for idx, s in zip(combo, signs):
                    v[idx] = s
                type_c.append(v)
        
        P = np.array(type_c, dtype=np.int64)
        n = len(P)
        
        if n < 220:
            return points_220
        
        # Compute dot products
        dots = P @ P.T
        
        # Conflict if dot > 2
        conflict = dots > 2
        np.fill_diagonal(conflict, False)
        
        # Greedy MIS: repeatedly select vertex with minimum remaining degree
        # Use vectorized operations for speed
        in_mis = np.zeros(n, dtype=bool)
        remaining = np.ones(n, dtype=bool)
        remaining_count = n
        
        # Precompute conflict as a list of lists for faster neighbor lookup
        # But for 5280 vertices, let's use numpy
        
        while remaining_count > 0 and time.time() < deadline:
            # Compute degrees: number of conflicts with remaining vertices
            degs = np.sum(conflict & remaining[None, :], axis=1)
            degs[~remaining] = np.inf
            
            v = int(np.argmin(degs))
            if not remaining[v]:
                break
            
            in_mis[v] = True
            remaining_count -= 1
            
            # Remove v and all its neighbors from remaining
            neighbors = conflict[v] & remaining
            remaining[v] = False
            remaining[neighbors] = False
            remaining_count -= np.sum(neighbors)
        
        mis_idx = np.where(in_mis)[0]
        result = P[mis_idx]
        
        if len(result) > len(points_220):
            # Verify the result
            if _is_valid(result):
                return result
        else:
            return points_220
            
    except Exception:
        pass
    
    # Fallback: return the 220-point solution
    return points_220

def _build_220(d):
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    return np.array(points, dtype=np.int64)

def _is_valid(points):
    if len(points) < 2:
        return False
    if points.shape[1] != 11:
        return False
    # Check no zero vector
    if np.any(np.all(points == 0, axis=1)):
        return False
    max_norm_sq = int(np.max(np.sum(points ** 2, axis=1)))
    min_dist_sq = float('inf')
    k = len(points)
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            dsq = int(np.dot(diff, diff))
            if dsq < min_dist_sq:
                min_dist_sq = dsq
            if min_dist_sq < max_norm_sq:
                return False
    return True
