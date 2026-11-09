import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    # Initial set: All vectors with squared norm 2 (minimal vectors of D11)
    # There are 2 * C(11, 2) = 220 such vectors.
    # This set is valid with min_dist_sq = 2 and max_norm_sq = 2.
    
    points = []
    for i, j in itertools.combinations(range(11), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(11, dtype=np.int64)
                v[i] = si
                v[j] = sj
                points.append(v)
    
    points = np.array(points)
    current_max_norm_sq = 2
    current_min_dist_sq = 2
    
    # Precompute candidate vectors for higher norms
    # We'll try to add vectors of norm 3, 4, etc.
    # A vector v can be added if:
    # 1. ||v||^2 >= current_min_dist_sq (Actually, the condition is min_dist_new >= max_norm_new)
    #    Let M = max(current_max_norm_sq, ||v||^2)
    #    We need dist(v, p) >= M for all p in points.
    #    Since we start with min_dist=2, and max_norm=2, if we add v with norm 2, M=2, need dist>=2.
    #    If we add v with norm 4, M=4, need dist>=4.
    
    # Let's generate candidates for norms 2, 3, 4, 5...
    # To be efficient, we'll generate candidates in order of increasing norm.
    # However, adding a high-norm vector is harder (needs larger distance).
    # So we should first exhaust all possible norm-2 additions (already done).
    # Then try norm-3?
    # Norm 3 vectors in Z^11:
    # Types:
    # 1. One entry +/- 1, one entry +/- 2? No, 1^2 + 2^2 = 5.
    # 2. Three entries +/- 1? 1+1+1 = 3.
    # So norm 3 vectors have three non-zero entries, each +/- 1.
    # Count: C(11, 3) * 2^3 = 165 * 8 = 1320.
    
    # Let's try to add norm-3 vectors.
    # Condition: For new v (norm 3), M = max(2, 3) = 3.
    # We need ||v - p||^2 >= 3 for all p in current points.
    # Current points have norm 2.
    # ||v - p||^2 = ||v||^2 + ||p||^2 - 2 v.p = 3 + 2 - 2 v.p = 5 - 2 v.p.
    # We need 5 - 2 v.p >= 3  => 2 v.p <= 2 => v.p <= 1.
    # So we need to find norm-3 vectors v such that their dot product with every existing norm-2 vector p is <= 1.
    
    # Let's generate all norm-3 vectors.
    candidates_norm3 = []
    for idxs in itertools.combinations(range(11), 3):
        i, j, k = idxs
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    v = np.zeros(11, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates_norm3.append(v)
    
    candidates_norm3 = np.array(candidates_norm3)
    
    # Greedy add:
    # Sort candidates? Or just iterate.
    # To maximize count, maybe we should prioritize vectors that are "far" from existing ones?
    # But the constraint is just a threshold.
    # Let's iterate through candidates and add if valid.
    
    added_any = True
    while added_any and time.time() < deadline - 10:
        added_any = False
        new_points = []
        for v in candidates_norm3:
            # Check distance to all current points
            # dist_sq = ||v-p||^2
            # We need min(dist_sq) >= 3
            # Efficient check:
            # diffs = v - points
            # dists_sq = np.sum(diffs**2, axis=1)
            # if np.min(dists_sq) >= 3: add
            # This is O(N * 11) per candidate. N=220+added. Candidates=1320.
            # 1320 * 220 * 11 ~ 3.2M ops. Fast.
            
            # Optimization: Use matrix multiplication?
            # dists_sq = ||v||^2 + ||p||^2 - 2 v.p
            # ||v||^2 = 3, ||p||^2 = 2
            # dists_sq = 5 - 2 * (v @ points.T)
            # We need 5 - 2 * dot >= 3 => dot <= 1.
            # So we just need to check if max(v @ points.T) <= 1.
            
            dots = v @ points.T
            if np.max(dots) <= 1:
                new_points.append(v)
        
        if new_points:
            # Add all valid new points?
            # Wait, adding one point changes the set, which might invalidate other candidates?
            # No, the condition is pairwise.
            # If v1 and v2 are both valid against the *original* set, are they valid against each other?
            # dist(v1, v2) must be >= max(norm(v1), norm(v2)) = 3.
            # We haven't checked v1-v2 distance yet.
            # So we should add them one by one, or check mutual distances.
            # Greedy: Add one, update set, re-evaluate remaining?
            # Or: Check if the set of new_points is internally valid and valid against old set.
            # Let's do one-by-one for simplicity and safety, but it's slower.
            # Actually, let's just add the first valid one found in this pass, then restart pass?
            # Or add all that are valid against current set, then check internal validity of the new batch?
            
            # Let's try adding them one by one in the order found, checking against current `points` (which grows).
            # This is O(C * N * 11) total if we don't restart the candidate list.
            # But if we add v1, v2 might become invalid if v2 is too close to v1.
            
            # Let's refine:
            # Iterate candidates. If valid against current `points`, add to `points`.
            # This ensures all constraints are met.
            
            for v in new_points:
                # Re-check against the *updated* points?
                # The `new_points` list was built against the `points` from the start of the loop.
                # If we add v1, then check v2 against points (including v1), it might fail.
                # So we should just do a single pass, adding immediately if valid.
                
                pass # We will restructure the loop below
            
            # Restructured approach:
            # Instead of collecting `new_points` and then adding, we should add immediately in the inner loop.
            # But `points` is an array. Appending to array is slow.
            # Let's use a list for points during the greedy phase.
            
            break # Break out of while loop to restructure with list
            
    # Restructure to use list for efficiency during greedy phase
    
    points_list = [p for p in points]
    current_max_norm_sq = 2
    
    # Candidates for norm 3
    candidates = candidates_norm3.tolist()
    
    # We can also try norm 4?
    # Norm 4 vectors:
    # 1. One entry +/- 2. (11 * 2 = 22 vectors)
    # 2. Two entries +/- 1, two entries +/- 1? No, 1+1+1+1=4. Four entries +/- 1.
    #    C(11, 4) * 2^4 = 330 * 16 = 5280 vectors.
    # 3. One entry +/- 1, one entry +/- 1, one entry +/- 1? No, sum is 3.
    #    Wait, 2^2 + 0... = 4.
    #    So types: (2,0..), (1,1,1,1,0..)
    
    # Let's stick to Norm 3 first as it's more likely to pack densely.
    
    # Greedy Loop
    # To allow adding multiple in one "round", we iterate through candidates.
    # If a candidate is valid against the *current* list, we add it.
    # We repeat this until no more can be added or time runs out.
    
    # Optimization: Pre-compute dot products?
    # As the list grows, we recompute.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # We iterate through a copy of candidates? Or the same list?
        # If we add a point, it might make other candidates invalid.
        # So we should check each candidate against the *current* state.
        
        # To avoid O(N^2) complexity in the worst case (if we re-scan all candidates for every addition),
        # we can just do a single pass. If we add some, we do another pass.
        
        for idx, v in enumerate(candidates):
            if time.time() > deadline - 5:
                break
            
            # Check validity against all points in points_list
            # v is norm 3.
            # For p in points_list:
            #   If ||p||^2 == 2: need v.p <= 1
            #   If ||p||^2 == 3: need ||v-p||^2 >= 3.
            #     ||v-p||^2 = 3 + 3 - 2 v.p = 6 - 2 v.p >= 3 => 2 v.p <= 3 => v.p <= 1.5 => v.p <= 1 (since integer).
            # So in both cases (norm 2 and norm 3 neighbors), the condition is v.p <= 1.
            # Wait, what if we add norm 4 later?
            # For now, all points have norm 2 or 3.
            # If p has norm 2: v.p <= 1.
            # If p has norm 3: v.p <= 1.
            
            # So we just need max(v @ p) <= 1 for all p in points_list.
            
            # Convert v to array for dot product
            v_arr = np.array(v)
            
            # Efficient check:
            # If points_list is large, converting to array each time is slow.
            # Let's maintain an array `points_arr` alongside the list?
            # Or just check in loop?
            # N is likely to stay under 1000. 1000 * 11 = 11000 ops. 1320 candidates * 11000 = 14M ops.
            # Python loop over 1320 candidates, inner loop over N points.
            # 1320 * 1000 = 1.3M iterations. In Python, this is ~0.1s - 1s. Acceptable.
            
            valid = True
            for p in points_list:
                dot = np.dot(v_arr, p)
                if dot > 1:
                    valid = False
                    break
            
            if valid:
                points_list.append(v)
                improved = True
                # Remove from candidates to avoid re-checking?
                # Or just let it be re-checked (it will be invalid now? No, it's in the set now).
                # Actually, if v is in the set, we don't need to check it again.
                # But `candidates` is a list of lists. We can mark it or just skip if it's in points_list?
                # Checking `if v in points_list` is O(N).
                # Better: Just let the loop finish. If it's already added, dot with itself is 3 > 1, so it will be rejected next time.
                # So no need to remove.
                
    # Convert back to array
    if points_list:
        final_points = np.array(points_list, dtype=np.int64)
    else:
        final_points = points # Fallback to initial set

    # Final Validation
    if final_points.shape[1] != 11 or final_points.shape[0] < 2:
        return np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.int64)

    # Check zero vector
    if np.any(np.all(final_points == 0, axis=1)):
        # Remove zero vectors if any (shouldn't happen)
        mask = ~np.all(final_points == 0, axis=1)
        final_points = final_points[mask]
        if final_points.shape[0] < 2:
             return np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ], dtype=np.int64)

    # Check min dist >= max norm
    k = final_points.shape[0]
    norms_sq = np.sum(final_points**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    min_dist_sq = float('inf')
    # Vectorized distance check for speed?
    # For k ~ 220+, O(k^2) is fine.
    for i in range(k):
        for j in range(i + 1, k):
            diff = final_points[i] - final_points[j]
            d_sq = np.sum(diff**2)
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
    
    if min_dist_sq < max_norm_sq:
        # Fallback to initial valid set
        return points.astype(np.int64)
        
    return np.around(final_points).astype(np.int64)

# Run once to ensure it's defined
if __name__ == "__main__":
    pass
