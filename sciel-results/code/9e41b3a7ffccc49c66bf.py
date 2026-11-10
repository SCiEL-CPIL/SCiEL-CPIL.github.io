# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1)
    
    # Precompute all 4-combinations of indices for the grid? No, that's huge.
    # We need to check subsets of the *selected* points.
    
    # Initial set: Randomly select a reasonably large set, say 20 points.
    # Or better, start with a greedy set to have a valid starting point.
    
    def get_valid_greedy_set(rng, target_size=15):
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.int64)
        remaining = list(range(N))
        rng.shuffle(remaining)
        
        # Simple greedy to get a valid starting set
        for idx in remaining:
            p = grid[idx]
            p_sq = grid_sq[idx]
            
            # Check against existing
            valid = True
            m = len(arr)
            if m >= 4:
                diff = arr - p
                sq_diff = arr_sq - p_sq
                w = np.column_stack((diff, sq_diff))
                # Check all 4-combos
                # Optimization: If m is small, just do it.
                combos = np.array(list(itertools.combinations(range(m), 4)))
                if len(combos) > 0:
                    sub = w[combos]
                    dets = np.linalg.det(sub)
                    if np.any(np.abs(dets) < 0.5):
                        valid = False
            
            if valid:
                arr = np.vstack([arr, p]) if len(arr) > 0 else p[None, :]
                arr_sq = np.append(arr_sq, p_sq)
                if len(arr) >= target_size:
                    break
        return arr, arr_sq

    def score_set(arr, arr_sq):
        """
        Compute a penalty score. Lower is better.
        Score = sum over all 5-point subsets of -log(|det|) if |det| < threshold, else 0.
        Actually, the condition is |det| >= 0.5.
        So if |det| < 0.5, it's invalid.
        We want to maximize the minimum |det|, or minimize the number of violations.
        Let's define score = sum( max(0, 0.5 - |det|) ) for all 5-tuples.
        This is 0 if valid.
        """
        m = len(arr)
        if m < 5:
            return 0.0
        
        total_penalty = 0.0
        # Iterate over all 5-point subsets
        # This is expensive. C(20, 5) = 15504. C(30, 5) = 142506.
        # We need to be fast.
        
        # Vectorized approach for a single 5-tuple is not easy without loops.
        # But we can precompute the matrix for a fixed p (the 5th point) against all 4-tuples of others?
        # No, the 5 points are arbitrary.
        
        # Let's just loop. It's the bottleneck.
        # To speed up, we can use the fact that if a 4-tuple is "bad" (determinant small with p),
        # we can stop early? No, we need the sum.
        
        # Alternative: Random sampling of 5-tuples? No, we need exact validity for the final answer.
        # But for the *score* during SA, we can use a heuristic or a subset of checks?
        # Risky. Let's stick to exact but optimize the inner loop.
        
        # Actually, for m~20-30, C(m,5) is manageable in Python if we vectorize the determinant calc?
        # np.linalg.det on a batch of matrices is fast.
        
        # Generate all 5-combinations of indices
        # This generation itself is slow in Python.
        
        # Let's try a different score:
        # For each point p_i, check against all 4-tuples of the *other* points.
        # Sum of penalties.
        
        # Precompute indices
        idxs = np.arange(m)
        
        # This is still O(m * C(m-1, 4))
        
        # Let's just do it.
        # To avoid generating huge arrays of combinations in Python,
        # we can iterate i, then generate combos of the remaining.
        
        # Optimization: If m is large, this is slow.
        # Let's limit the SA to m <= 25 initially.
        
        penalty = 0.0
        # We can parallelize? No.
        
        # Let's use a precomputed list of all 5-combinations of range(m)?
        # If m changes, we regenerate.
        
        # For the sake of speed in the SA loop, let's approximate the score
        # by checking a random sample of 5-tuples if m is large,
        # or use a faster metric.
        
        # Actually, let's just count the number of violations.
        # If we find any violation, penalty is high.
        
        # Let's try to vectorize the check for a specific i.
        # Fix i. p = arr[i].
        # Others = np.delete(arr, i, axis=0)
        # Others_sq = np.delete(arr_sq, i)
        # Generate 4-combos of others.
        
        # This is still slow in pure Python/NumPy for many i.
        
        # Let's change strategy for the score:
        # Use a random subset of 5-tuples to estimate the "badness".
        # If we find a bad one, return a high penalty.
        # This makes the score noisy but fast.
        
        num_samples = 100
        if m < 5:
            return 0.0
            
        # Randomly sample 5-tuples
        # We can just pick random indices
        indices = np.random.choice(m, size=(num_samples, 5), replace=False)
        
        # For each sampled tuple, compute det
        # This requires gathering points.
        # p1 = arr[indices[:, 0]], etc.
        
        p1 = arr[indices[:, 0]]
        p2 = arr[indices[:, 1]]
        p3 = arr[indices[:, 2]]
        p4 = arr[indices[:, 3]]
        p5 = arr[indices[:, 4]]
        
        # Matrix rows: (q - p, |q|^2 - |p|^2)
        # We need to form 4x4 matrices for each sample.
        # Rows are q1-p5, q2-p5, q3-p5, q4-p5.
        # And the sq diffs.
        
        # diff1 = p1 - p5, sq1 = arr_sq[indices[:, 0]] - arr_sq[indices[:, 4]]
        # ...
        
        d1 = p1 - p5
        d2 = p2 - p5
        d3 = p3 - p5
        d4 = p4 - p5
        
        s1 = arr_sq[indices[:, 0]] - arr_sq[indices[:, 4]]
        s2 = arr_sq[indices[:, 1]] - arr_sq[indices[:, 4]]
        s3 = arr_sq[indices[:, 2]] - arr_sq[indices[:, 4]]
        s4 = arr_sq[indices[:, 3]] - arr_sq[indices[:, 4]]
        
        # Stack to form (num_samples, 4, 4)
        # Row 1: d1[:, 0], d1[:, 1], d1[:, 2], s1
        # This is a bit awkward to vectorize into a 3D array directly.
        
        # Let's construct the matrices.
        # M[:, 0, :] = [d1[:, 0], d1[:, 1], d1[:, 2], s1]
        # M[:, 1, :] = [d2[:, 0], d2[:, 1], d2[:, 2], s2]
        # M[:, 2, :] = [d3[:, 0], d3[:, 1], d3[:, 2], s3]
        # M[:, 3, :] = [d4[:, 0], d4[:, 1], d4[:, 2], s4]
        
        M = np.zeros((num_samples, 4, 4))
        M[:, 0, 0] = d1[:, 0]
        M[:, 0, 1] = d1[:, 1]
        M[:, 0, 2] = d1[:, 2]
        M[:, 0, 3] = s1
        
        M[:, 1, 0] = d2[:, 0]
        M[:, 1, 1] = d2[:, 1]
        M[:, 1, 2] = d2[:, 2]
        M[:, 1, 3] = s2
        
        M[:, 2, 0] = d3[:, 0]
        M[:, 2, 1] = d3[:, 1]
        M[:, 2, 2] = d3[:, 2]
        M[:, 2, 3] = s3
        
        M[:, 3, 0] = d4[:, 0]
        M[:, 3, 1] = d4[:, 1]
        M[:, 3, 2] = d4[:, 2]
        M[:, 3, 3] = s4
        
        dets = np.linalg.det(M)
        abs_dets = np.abs(dets)
        
        # Penalty: sum of max(0, 0.5 - abs_det)
        violations = np.maximum(0, 0.5 - abs_dets)
        penalty = np.sum(violations)
        
        # If we found any violation, the score is bad.
        # To make it more sensitive, we can return the min abs_det?
        # Or just the sum of violations.
        
        return penalty

    def exact_validity_check(arr, arr_sq):
        m = len(arr)
        if m < 5:
            return True
        
        # Exact check for the final result
        # This must be robust.
        # Use the same vectorized approach but with ALL combinations.
        # C(m, 5) can be large.
        # If m=25, C(25,5) = 53130.
        # If m=30, C(30,5) = 142506.
        # This is doable if we batch it.
        
        # Generate all 5-combos
        # This is the slow part.
        # Let's use itertools in a loop but vectorize the det calc in chunks?
        # Or just do it all at once if memory allows.
        
        # For m=30, 142k matrices of 4x4. 142k * 16 * 8 bytes = ~18 MB. Fine.
        
        try:
            combos = np.array(list(itertools.combinations(range(m), 5)))
        except MemoryError:
            return False # Fallback, shouldn't happen for m<50
            
        if len(combos) == 0:
            return True
            
        # Extract points
        p1 = arr[combos[:, 0]]
        p2 = arr[combos[:, 1]]
        p3 = arr[combos[:, 2]]
        p4 = arr[combos[:, 3]]
        p5 = arr[combos[:, 4]]
        
        d1 = p1 - p5
        d2 = p2 - p5
        d3 = p3 - p5
        d4 = p4 - p5
        
        s1 = arr_sq[combos[:, 0]] - arr_sq[combos[:, 4]]
        s2 = arr_sq[combos[:, 1]] - arr_sq[combos[:, 4]]
        s3 = arr_sq[combos[:, 2]] - arr_sq[combos[:, 4]]
        s4 = arr_sq[combos[:, 3]] - arr_sq[combos[:, 4]]
        
        M = np.zeros((len(combos), 4, 4))
        M[:, 0, 0] = d1[:, 0]
        M[:, 0, 1] = d1[:, 1]
        M[:, 0, 2] = d1[:, 2]
        M[:, 0, 3] = s1
        
        M[:, 1, 0] = d2[:, 0]
        M[:, 1, 1] = d2[:, 1]
        M[:, 1, 2] = d2[:, 2]
        M[:, 1, 3] = s2
        
        M[:, 2, 0] = d3[:, 0]
        M[:, 2, 1] = d3[:, 1]
        M[:, 2, 2] = d3[:, 2]
        M[:, 2, 3] = s3
        
        M[:, 3, 0] = d4[:, 0]
        M[:, 3, 1] = d4[:, 1]
        M[:, 3, 2] = d4[:, 2]
        M[:, 3, 3] = s4
        
        dets = np.linalg.det(M)
        abs_dets = np.abs(dets)
        
        return np.all(abs_dets >= 0.5)

    def sa_search(initial_arr, initial_sq, start_time, deadline):
        current_arr = initial_arr.copy()
        current_sq = initial_sq.copy()
        current_score = score_set(current_arr, current_sq)
        
        best_arr = current_arr.copy()
        best_sq = current_sq.copy()
        best_score = current_score
        
        T = 1.0
        cooling_rate = 0.999
        min_T = 1e-4
        
        # We will try to add points or swap points.
        # State: set of indices.
        # Let's represent state as a boolean mask or a set of indices.
        # Since we need to return the points, let's keep the array.
        
        # To allow adding points, we need to consider points NOT in the set.
        # This is complex with a fixed-size array.
        # Let's fix the size K for the SA phase?
        # Or allow size to change?
        
        # Strategy:
        # 1. Start with a valid set of size K0.
        # 2. SA loop:
        #    - With prob 0.5, try to swap a point in the set with a point outside.
        #    - With prob 0.5, try to add a point (if size < max_size).
        #    - With prob 0.5, try to remove a point (if size > min_size).
        
        # This variable size is tricky for the score function which assumes fixed m.
        # But score_set handles variable m.
        
        # Let's stick to fixed size K for simplicity in SA, then try to expand?
        # No, greedy expansion is bad.
        # Let's allow size changes.
        
        # To make it efficient, we need to update the score incrementally?
        # Hard with random sampling score.
        # Let's just recalculate.
        
        # Max size 25, Min size 10.
        max_size = 25
        min_size = 10
        
        # Precompute list of all indices
        all_indices = np.arange(N)
        
        while time.time() < deadline and T > min_T:
            m = len(current_arr)
            
            # Generate neighbor
            if np.random.rand() < 0.3 and m < max_size:
                # Add a point
                candidates = np.setdiff1d(all_indices, current_arr)
                if len(candidates) == 0: continue
                idx = np.random.choice(candidates)
                new_arr = np.vstack([current_arr, grid[idx]])
                new_sq = np.append(current_sq, grid_sq[idx])
            elif np.random.rand() < 0.3 and m > min_size:
                # Remove a point
                idx_to_remove = np.random.choice(m)
                new_arr = np.delete(current_arr, idx_to_remove, axis=0)
                new_sq = np.delete(current_sq, idx_to_remove)
            else:
                # Swap
                if m == 0: continue
                idx_in = np.random.choice(m)
                candidates = np.setdiff1d(all_indices, current_arr)
                if len(candidates) == 0: continue
                idx_out = np.random.choice(candidates)
                
                new_arr = current_arr.copy()
                new_sq = current_sq.copy()
                new_arr[idx_in] = grid[idx_out]
                new_sq[idx_in] = grid_sq[idx_out]
                
                # Ensure no duplicates in new_arr?
                # Since we picked idx_out from candidates (not in current),
                # and we replaced one element, there are no duplicates.
            
            new_score = score_set(new_arr, new_sq)
            
            delta = new_score - current_score
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_arr = new_arr
                current_sq = new_sq
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_arr = current_arr.copy()
                    best_sq = current_sq.copy()
            
            T *= cooling_rate
            
            # Periodic check for exact validity of best_arr to ensure we don't return invalid
            # And maybe restart if stuck?
            
        return best_arr, best_sq

    start_time = time.time()
    deadline = start_time + 600
    
    rng = np.random.default_rng(42)
    
    # Initial set
    init_arr, init_sq = get_valid_greedy_set(rng, target_size=15)
    
    best_overall_arr, best_overall_sq = sa_search(init_arr, init_sq, start_time, deadline)
    
    # Try a few more restarts if time permits
    while time.time() < deadline - 60:
        init_arr, init_sq = get_valid_greedy_set(rng, target_size=15)
        cand_arr, cand_sq = sa_search(init_arr, init_sq, start_time, deadline)
        
        # Exact check candidate
        if exact_validity_check(cand_arr, cand_sq):
            if len(cand_arr) > len(best_overall_arr):
                best_overall_arr = cand_arr
                best_overall_sq = cand_sq
        else:
            # If SA found an invalid set, try to repair it?
            # Or just discard.
            # Let's try to repair by removing points until valid.
            temp_arr = cand_arr.copy()
            temp_sq = cand_sq.copy()
            # Greedy removal of points involved in violations?
            # Too complex. Just discard.
            pass
            
    # Final exact check
    if not exact_validity_check(best_overall_arr, best_overall_sq):
        # Fallback to a known valid small set
        # Just return the first 10 points from grid, which should be valid?
        # No, let's just return a small valid set.
        # Greedy with size 5 is always valid.
        return init_arr[:5] if len(init_arr) >= 5 else init_arr
    
    return best_overall_arr.astype(np.int64)

# EVOLVE-BLOCK-END
