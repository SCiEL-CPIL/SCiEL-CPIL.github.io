import numpy as np
import itertools
import time

def generate_vectors_up_to_norm2(d=11):
    """
    Generate all integer vectors in Z^d with squared norm 1 and 2.
    Norm 1: e_i (11 vectors)
    Norm 2: e_i +/- e_j (2 * C(11,2) = 110 vectors)
    Total: 121 vectors.
    """
    vecs = []
    # Norm 1
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 1
        vecs.append(v)
    # Norm 2
    for idxs in itertools.combinations(range(d), 2):
        i, j = idxs
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                vecs.append(v)
    return np.array(vecs)

def generate_vectors_norm4(d=11):
    """
    Generate all integer vectors in Z^d with squared norm 4.
    Types:
    1. (2, 0, ...) : 2^2 + 0 = 4. Count: C(11,1) * 2 = 22
    2. (1, 1, 1, 1, ...) : 1^2+1^2+1^2+1^2 = 4. Count: C(11,4) * 2^4 = 330 * 16 = 5280
    Total: 5302 vectors.
    """
    vecs = []
    # Type 1: One coordinate is +/- 2
    for i in range(d):
        for s in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = 2 * s
            vecs.append(v)
    
    # Type 2: Four coordinates are +/- 1
    for idxs in itertools.combinations(range(d), 4):
        i, j, k, l = idxs
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    for s4 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        v[k] = s3
                        v[l] = s4
                        vecs.append(v)
    return np.array(vecs)

def validate_configuration(points):
    """
    Validate the configuration according to the task contract.
    Returns True if valid, False otherwise.
    """
    if points.shape[1] != 11:
        return False
    k = points.shape[0]
    if k < 2:
        return False
    
    # Check no zero vector
    if np.any(np.all(points == 0, axis=1)):
        return False
        
    # Calculate max squared norm
    norms_sq = np.sum(points**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Calculate min squared distance
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            d_sq = np.sum(diff**2)
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                
    # Condition: min_dist_sq >= max_norm_sq
    return min_dist_sq >= max_norm_sq

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    # Initial set: All vectors with norm^2 = 2.
    # This is the standard D11 minimal vector set.
    # K=220. Min Dist = 2, Max Norm = 2. Valid.
    initial_vecs = generate_vectors_up_to_norm2(11)
    # Filter to only norm 2 for the base set to keep Max Norm = 2 initially?
    # Actually, if we include norm 1, Max Norm becomes 2 (from norm 2 vecs).
    # Dist(e1, e1+e2) = 1. 1 < 2. Invalid.
    # So we cannot mix norm 1 and norm 2.
    # The seed program used only norm 2. We stick to that as the base.
    
    base_vecs = []
    for idxs in itertools.combinations(range(11), 2):
        i, j = idxs
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(11, dtype=np.int64)
                v[i] = si
                v[j] = sj
                base_vecs.append(v)
    
    current_points = np.array(base_vecs)
    
    # Calculate current properties
    current_norms_sq = np.sum(current_points**2, axis=1)
    current_max_norm_sq = np.max(current_norms_sq) # Should be 2
    
    # Candidate pool: Vectors with norm^2 = 4.
    # If we add a vector with norm^2 = 4, Max Norm becomes 4.
    # We must ensure Min Dist >= 4.
    # Current Min Dist is 2.
    # If we add ANY vector, the new Min Dist will be min(Old Min Dist, Dist(New, Old)).
    # So we need Dist(New, Old) >= 4 for ALL Old points.
    # AND we need Dist(New, Other New) >= 4.
    
    candidates = generate_vectors_norm4(11)
    
    # Optimization: Pre-compute norms of candidates (all are 4)
    # We need to check distance against all current points.
    
    new_points = []
    
    # To speed up, we can iterate through candidates and check if they are "far enough"
    # from all existing points (base + newly added).
    
    # Since checking against 220 base points for 5302 candidates is 220*5302 ~ 1.2M ops,
    # which is fast in NumPy if vectorized, or even Python loop if simple.
    # But we also need to check against newly added points.
    
    # Let's use a greedy approach.
    # Sort candidates by some heuristic? Random? Or just iterate.
    # Since we want to maximize count, order might not matter much if we just add if valid.
    # But adding one might block others.
    
    # Let's try to add as many as possible.
    
    # Convert current points to a list for easy appending and indexing
    selected = [p for p in current_points]
    
    # Pre-compute squared norms of selected (all 2 initially)
    # We will maintain a list of selected points.
    
    # For each candidate:
    # 1. Check distance to all selected points >= 4.
    # 2. If yes, add to selected.
    
    # To make this faster, we can use a vectorized check against the current selected set.
    # However, the selected set grows.
    
    # Let's estimate time.
    # 5302 candidates.
    # If we check each against ~220 base points: 1.2M distance calculations.
    # Each distance calc is 11 subtractions, 11 squares, 11 sums. ~33 ops.
    # 1.2M * 33 = 40M ops. Python is slow, NumPy is fast.
    # Using NumPy:
    # selected_arr = np.array(selected)
    # diff = candidate[None, :] - selected_arr
    # dists = np.sum(diff**2, axis=1)
    # if np.min(dists) >= 4: add
    
    # As selected grows, this gets slower.
    # If we add 1000 points, checking 1000 distances per candidate.
    # 5302 * 1000 = 5.3M vector ops of size 11.
    # This should be well within 580s.
    
    selected_arr = current_points.copy()
    
    for i in range(len(candidates)):
        if time.time() > deadline:
            break
            
        cand = candidates[i]
        
        # Check distance to all selected points
        # Vectorized difference
        diffs = selected_arr - cand
        dists_sq = np.sum(diffs**2, axis=1)
        
        if np.min(dists_sq) >= current_max_norm_sq:
            # Note: current_max_norm_sq is 4 if we've added any norm 4 vector.
            # Initially 2.
            # Wait, if we add a norm 4 vector, the max norm of the *set* becomes 4.
            # The condition is min_dist >= max_norm.
            # If we add a norm 4 vector, max_norm becomes 4.
            # So we must ensure ALL pairwise distances are >= 4.
            # This includes distances between old norm-2 vectors.
            # Old min dist was 2.
            # 2 < 4.
            # So the configuration becomes INVALID immediately upon adding the first norm-4 vector!
            # Because the old pairs still have distance 2.
            
            # Therefore, we CANNOT simply add norm-4 vectors to the norm-2 set.
            # We must REPLACE the norm-2 set or find a set of norm-4 vectors that are mutually far apart,
            # AND ensure that if we keep any norm-2 vectors, the distances between them are >= 4.
            # But norm-2 vectors have min dist 2.
            # So we cannot have ANY norm-2 vectors if Max Norm is 4.
            
            # Conclusion: We must choose a homogeneous norm set.
            # Option A: Norm 2. K=220.
            # Option B: Norm 4. Find max subset of norm-4 vectors with min dist >= 4.
            # Option C: Norm 9 (3^2). etc.
            
            # Let's pivot the strategy to search for a large set of Norm 4 vectors.
            # This is a packing problem in the set of 5302 vectors.
            # Constraint: ||v_i - v_j||^2 >= 4.
            
            # Let's restart the search logic for Norm 4.
            pass
            
    # The logic above revealed a flaw in the "add to existing" strategy.
    # We need to search for a large independent set in the graph of Norm 4 vectors,
    # where edges connect vectors with distance < 4.
    # Since all have norm 4, dist^2 = 8 - 2*v_i.v_j.
    # dist^2 >= 4  => 8 - 2*v_i.v_j >= 4 => 4 >= 2*v_i.v_j => v_i.v_j <= 2.
    
    # So we want a set of norm-4 vectors with pairwise dot products <= 2.
    
    # Let's implement a greedy search for this specific constraint.
    
    # Re-initialize for Norm 4 search
    selected_norm4 = []
    
    # To improve greedy, we might want to randomize the order of candidates or use a heuristic.
    # Randomizing order can help escape local minima in greedy set packing.
    # Let's do a few random restarts if time permits.
    
    best_k = 0
    best_points = current_points # Fallback to Norm 2 set (K=220)
    
    # We know Norm 2 gives 220.
    # Let's see if we can beat 220 with Norm 4.
    
    # Candidate list for Norm 4
    cand_list = candidates.tolist()
    
    # Number of restarts based on time
    # Each restart: shuffle candidates, greedy add.
    # Greedy add complexity: O(N * K * D) where N is candidates, K is selected, D is dim.
    # N=5302, K~?, D=11.
    
    max_restarts = 50 # Estimate
    restart_count = 0
    
    while restart_count < max_restarts:
        if time.time() > deadline:
            break
            
        np.random.shuffle(cand_list)
        
        current_sel = []
        
        for cand in cand_list:
            if time.time() > deadline:
                break
                
            # Check dot product constraint: v.c <= 2
            # For all v in current_sel
            
            valid = True
            for v in current_sel:
                dot = np.dot(cand, v)
                if dot > 2:
                    valid = False
                    break
            
            if valid:
                current_sel.append(cand)
                
        k_sel = len(current_sel)
        if k_sel > best_k:
            best_k = k_sel
            best_points = np.array(current_sel)
            
        restart_count += 1
        
    # Final check: if best_k is 0 or less than 220, return the Norm 2 set.
    # If best_k > 220, return the Norm 4 set.
    
    if best_k > 220:
        points = np.around(best_points).astype(np.int64)
        # Validate just in case
        if validate_configuration(points):
            return points
    
    # Fallback to Norm 2
    points = np.around(current_points).astype(np.int64)
    if validate_configuration(points):
        return points
        
    # Absolute fallback
    fallback = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int64)
    return fallback

# Optional: Try to improve further if time permits?
# Given the structural nature, the norm-2 set is a hard bound for "minimal vectors" in D11.
# Searching for larger sets in other norms is risky and computationally heavy.
# The prompt suggests a structural change. Exact construction of D11 minimal vectors is the most robust improvement over greedy search on N=3/4.

# Note: Can we add vectors of norm 1?
# If we add e1 (norm 1), max norm becomes 2.
# Dist(e1, e1+e2) = 1. 1 < 2. Fails.
# So we cannot mix norm 1 and norm 2.
# Can we mix norm 2 and norm 4?
# If we add 2*e1 (norm 4), max norm 4.
# Dist(2*e1, e1+e2) = (1, -1) -> 2. 2 < 4. Fails.
# So mixing norms generally breaks the condition because the larger norm vector gets "too close" to the smaller ones.
# Thus, the optimal sets are likely homogeneous in norm.
# Norm 1: k=22.
# Norm 2: k=220.
# Norm 3: Need to solve a coding theory problem. Unlikely to exceed 220 quickly.
# Norm 4: Even sparser.

# Therefore, k=220 is likely the global optimum or very close to it for integer lattices in this setting.
