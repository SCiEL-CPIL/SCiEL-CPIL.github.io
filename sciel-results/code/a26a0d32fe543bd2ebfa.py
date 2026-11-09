# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time
import random

def kissing_number11() -> np.ndarray:
    """
    Attempts to maximize the number of integer points in 11 dimensions
    satisfying the kissing constraint using a Greedy Local Search strategy.
    """
    start_time = time.time()
    deadline = start_time + 620  # 620s to leave 10s margin
    
    d = 11
    
    # 1. Generate the base set of 220 weight-2 vectors
    base_points = []
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                base_points.append(p)
    
    current_points = np.array(base_points, dtype=np.int64)
    k = current_points.shape[0]
    
    # Helper to check validity
    def is_valid(points):
        if points.shape[0] < 2:
            return False
        norms_sq = np.sum(points**2, axis=1)
        if np.any(norms_sq == 0):
            return False
        max_norm_sq = np.max(norms_sq)
        # Check min dist
        # For speed, we can use vectorized ops for small k, but k~220 is small enough for loops if optimized
        # However, the evaluator uses a Python loop. We should mimic that or be faster.
        # We just need to ensure OUR check is correct.
        min_dist_sq = float('inf')
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                diff = points[i] - points[j]
                dist_sq = int(np.sum(diff**2))
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                if min_dist_sq < max_norm_sq:
                    return False
        return min_dist_sq >= max_norm_sq

    # Verify base set
    if not is_valid(current_points):
        # Fallback
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    best_points = current_points.copy()
    best_k = k
    
    # 2. Generate candidate pool for swaps
    # Include weight 1, 2, 3, 4 vectors
    candidates = []
    # Weight 1
    for i in range(d):
        for s in [1, -1]:
            p = np.zeros(d, dtype=np.int64)
            p[i] = s
            candidates.append(p)
    # Weight 2 (already in base, but include for completeness if we remove and re-add)
    # Weight 3
    for comb in combinations(range(d), 3):
        for mask in range(8):
            p = np.zeros(d, dtype=np.int64)
            for idx, c in enumerate(comb):
                p[c] = 1 if (mask >> idx) & 1 else -1
            candidates.append(p)
    # Weight 4
    for comb in combinations(range(d), 4):
        for mask in range(16):
            p = np.zeros(d, dtype=np.int64)
            for idx, c in enumerate(comb):
                p[c] = 1 if (mask >> idx) & 1 else -1
            candidates.append(p)
            
    candidates = np.array(candidates, dtype=np.int64)
    
    # 3. Local Search
    # We try to swap a point from current_points with a point from candidates
    # or just add a point if possible (though space is tight)
    # Since we start at a local maximum of weight-2, we likely can't add.
    # We try swaps: remove one, add one.
    
    # To speed up validity check, we can precompute norms
    # But for simplicity, we run is_valid.
    
    # Optimization: Only try swaps that might improve or maintain k.
    # Actually, we want to find k > 220.
    # Swaps keep k=220.
    # Can we add a point?
    # If we can add a point to the 220 set, we get 221.
    # Let's check if any candidate can be added to the base set.
    
    # Check addability
    base_set_list = list(base_points)
    # This is O(k * m) which is 220 * 5000 = 1.1M ops. Fast.
    
    # Precompute norms of base points
    base_norms = np.sum(np.array(base_points)**2, axis=1)
    max_base_norm = np.max(base_norms)
    
    # For each candidate, check if it can be added to base_points
    # Constraint: dist to all base points >= max(max_base_norm, norm(cand))
    # Since base points are weight 2, max_base_norm = 2.
    # If cand is weight 1, norm=1. Max norm = 2.
    # If cand is weight 3, norm=3. Max norm = 3.
    
    # We will try to add points to the current set.
    # Start with base 220.
    current_list = [p.copy() for p in base_points]
    current_norms = [2] * 220
    current_max_norm = 2
    
    added_any = False
    # Try adding weight 1 points
    # We need to check against all current points.
    # This is expensive if we do it for all 22 weight 1 points against 220 points.
    # 22 * 220 = 4840 distance checks. Fast.
    
    for cand in candidates:
        if time.time() > deadline - 10:
            break
            
        cand_norm = int(np.sum(cand**2))
        new_max_norm = max(current_max_norm, cand_norm)
        
        # Check if cand is already in current_list
        # Simple check: if cand_norm == 2 and cand in base_points, skip?
        # We are trying to ADD, so if it's already there, we can't add it again.
        # We assume candidates are unique? No, candidates list has duplicates?
        # Weight 2 candidates are in base.
        # We should only try to add if it's not already in the set.
        # To be safe, we check distance > 0.
        
        valid = True
        for p in current_list:
            diff = cand - p
            dist_sq = int(np.sum(diff**2))
            if dist_sq < new_max_norm:
                valid = False
                break
        
        if valid:
            # Add it
            current_list.append(cand)
            current_norms.append(cand_norm)
            current_max_norm = new_max_norm
            added_any = True
            # Reset candidates? Or just continue?
            # If we add one, the set changes, so previous "invalid" might become valid?
            # No, adding a point only makes constraints TIGHTER (max norm might increase, min dist might decrease).
            # So if it was invalid before, it will remain invalid.
            # We can just continue checking other candidates.
            
    if added_any:
        best_points = np.array(current_list, dtype=np.int64)
        best_k = len(current_list)
    else:
        best_points = current_points
        best_k = k

    # If we have time, try swaps to see if we can enable additions?
    # Unlikely.
    
    # Return the best found
    return np.around(best_points).astype(np.int64)

# EVOLVE-BLOCK-END
