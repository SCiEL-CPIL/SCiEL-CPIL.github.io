# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates 
    such that their maximum squared norm is less than or equal to their minimum pairwise squared distance.
    
    Strategy:
    1. Start with the set of all integer vectors with exactly two non-zero entries (±1).
       There are C(11,2) * 2^2 = 55 * 4 = 220 such points.
       All have squared norm 2.
       For any distinct p, q in this set:
       - If they share one index: dot product is ±1. dist^2 = 2+2-2(±1) = 4 or 0? 
         Wait. p=(1,1,0..), q=(1,-1,0..). diff=(0,2,0..). dist^2=4.
         p=(1,1,0..), q=(-1,1,0..). diff=(2,0,0..). dist^2=4.
         p=(1,1,0..), q=(1,1,0..) is same point, excluded.
       - If they share no indices: dot product is 0. dist^2 = 2+2-0 = 4.
       So min pairwise squared distance is 4.
       Max squared norm is 2.
       Condition: 4 >= 2. Valid.
       
    2. Try to add vectors with exactly four non-zero entries (±1).
       There are C(11,4) * 2^4 = 330 * 16 = 5280 such points.
       All have squared norm 4.
       If we add any such vector v, the new max squared norm becomes 4.
       We must ensure that for all pairs (u, w) in the new set, dist^2 >= 4.
       
       Check constraints:
       a) Between two 2-norm vectors (u, w): dist^2 = 4 (already verified). OK.
       b) Between two 4-norm vectors (v1, v2):
          dist^2 = ||v1||^2 + ||v2||^2 - 2 v1.v2 = 8 - 2 v1.v2.
          We need 8 - 2 v1.v2 >= 4 => 2 v1.v2 <= 4 => v1.v2 <= 2.
          The dot product of two 4-norm vectors with ±1 entries depends on the overlap of their supports and the signs.
          If supports are disjoint, dot = 0.
          If they overlap in k positions (k=1,2,3,4), the dot product is sum of products of signs in overlap.
          Max possible dot product is 4 (if identical). Min is -4.
          We need to select a subset of 4-norm vectors such that any pair has dot product <= 2.
          
       c) Between a 2-norm vector u and a 4-norm vector v:
          dist^2 = 2 + 4 - 2 u.v = 6 - 2 u.v.
          We need 6 - 2 u.v >= 4 => 2 u.v <= 2 => u.v <= 1.
          
    3. Greedy construction:
       - Initialize S with all 220 two-norm vectors.
       - Generate all 5280 four-norm vectors.
       - Sort them or iterate through them.
       - For each candidate v, check if it can be added:
         * Check dot product with all existing 4-norm vectors in S: must be <= 2.
         * Check dot product with all existing 2-norm vectors in S: must be <= 1.
       - If valid, add to S.
       
    4. Optimization:
       - Pre-compute dot products? No, just use numpy for speed.
       - Maintain separate lists for 2-norm and 4-norm vectors to avoid checking norms repeatedly.
       - Use numpy arrays for batch dot product checks.
       
    5. Time management:
       - Deadline: 590 seconds.
       - If time runs out, return the current set.
       
    Note: The problem asks for min_dist^2 >= max_norm^2.
    If we only use 2-norm vectors, max_norm^2 = 2, min_dist^2 = 4. 4 >= 2.
    If we add 4-norm vectors, max_norm^2 = 4. We must ensure min_dist^2 >= 4.
    The greedy approach ensures this for the selected subset.
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Generate all 2-norm vectors (two non-zeros, ±1)
    two_norm_vecs = []
    for i in range(d):
        for j in range(i + 1, d):
            # Four combinations of signs
            p1 = np.zeros(d, dtype=np.int64); p1[i] = 1; p1[j] = 1
            two_norm_vecs.append(p1)
            p2 = np.zeros(d, dtype=np.int64); p2[i] = -1; p2[j] = -1
            two_norm_vecs.append(p2)
            p3 = np.zeros(d, dtype=np.int64); p3[i] = 1; p3[j] = -1
            two_norm_vecs.append(p3)
            p4 = np.zeros(d, dtype=np.int64); p4[i] = -1; p4[j] = 1
            two_norm_vecs.append(p4)
    
    two_norm_arr = np.array(two_norm_vecs, dtype=np.int64)
    n_two = len(two_norm_arr)
    
    # Generate all 4-norm vectors (four non-zeros, ±1)
    four_norm_vecs = []
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for idx, val in zip(combo, signs):
                v[idx] = val
            four_norm_vecs.append(v)
    
    four_norm_arr = np.array(four_norm_vecs, dtype=np.int64)
    n_four = len(four_norm_arr)
    
    # We will greedily select a subset of four_norm_vecs.
    # selected_four_indices = []
    # For efficiency, we can maintain a numpy array of selected four-norm vectors.
    # But appending to numpy is slow. Let's use a list and convert to array when checking.
    # Or better: maintain a boolean mask or just a list of indices.
    
    # To check a candidate v against existing selected four-norms:
    # We need max dot product <= 2.
    # To check against all two-norms:
    # We need max dot product <= 1.
    
    # Pre-compute dot products of all four-norms with all two-norms?
    # n_four * n_two * d = 5280 * 220 * 11 ~ 12.8M ops. This is fast in numpy.
    # dots_4_2 = four_norm_arr @ two_norm_arr.T  # Shape (n_four, n_two)
    # This takes ~100ms maybe. Worth it.
    
    # Also pre-compute dot products of all four-norms with all four-norms?
    # n_four * n_four * d = 5280 * 5280 * 11 ~ 310M ops.
    # This might be ~1-2 seconds in numpy. Also worth it if it simplifies the loop.
    # dots_4_4 = four_norm_arr @ four_norm_arr.T # Shape (n_four, n_four)
    
    # Let's do the pre-computation.
    if time.time() > deadline - 10:
        return two_norm_arr
        
    try:
        # dots_4_2[i, j] = four_norm_vecs[i] dot two_norm_vecs[j]
        dots_4_2 = four_norm_arr @ two_norm_arr.T
        # dots_4_4[i, j] = four_norm_vecs[i] dot four_norm_vecs[j]
        dots_4_4 = four_norm_arr @ four_norm_arr.T
    except MemoryError:
        # Fallback to on-the-fly calculation if memory is an issue (unlikely for this size)
        dots_4_2 = None
        dots_4_4 = None

    # Greedy selection of 4-norm vectors
    # We want to maximize the number of selected 4-norm vectors.
    # Order of selection can matter. Let's try sorting by "conflict potential" or just random.
    # A vector with many high dot products with others is "conflict-prone".
    # Let's just iterate in original order for determinism, or shuffle for variety?
    # Original order is fine.
    
    selected_four = []
    
    # For fast checking, maintain a numpy array of selected four-norm vectors?
    # Or just indices.
    # If we have indices, we can slice the pre-computed dot matrix.
    # dots_4_4[selected_indices, :] gives dots with all candidates.
    # We need to check against selected indices only.
    
    # Let's maintain a list of selected indices.
    sel_idx = []
    
    for i in range(n_four):
        if time.time() > deadline - 5:
            break
            
        # Check against two-norms
        # Condition: max(dots_4_2[i, :]) <= 1
        if dots_4_2 is not None:
            if np.max(dots_4_2[i, :]) > 1:
                continue
        else:
            # On the fly
            if np.max(four_norm_arr[i] @ two_norm_arr.T) > 1:
                continue
                
        # Check against selected four-norms
        if sel_idx:
            if dots_4_4 is not None:
                # Get dot products with selected
                # dots_4_4[i, sel_idx]
                # We need max <= 2
                # Note: dots_4_4[i, i] = 4 (norm squared). But i is not in sel_idx yet.
                dots_with_sel = dots_4_4[i, sel_idx]
                if np.max(dots_with_sel) > 2:
                    continue
            else:
                # On the fly
                sel_vecs = four_norm_arr[sel_idx]
                dots_with_sel = sel_vecs @ four_norm_arr[i]
                if np.max(dots_with_sel) > 2:
                    continue
        else:
            # No selected four-norms yet, always valid (assuming passed two-norm check)
            pass
            
        # Add to selected
        sel_idx.append(i)
        
    # Construct final array
    # Combine two_norm_arr and selected four_norms
    if sel_idx:
        selected_four_arr = four_norm_arr[sel_idx]
        final_points = np.vstack([two_norm_arr, selected_four_arr])
    else:
        final_points = two_norm_arr
        
    # Final verification (optional but recommended for safety)
    # The problem requires min_dist^2 >= max_norm^2.
    # Max norm squared is 4 (if any 4-norm selected) or 2 (if not).
    # Min dist squared should be >= 4 or 2 respectively.
    # Our greedy logic ensures:
    # - 2-2 pairs: dist^2 = 4.
    # - 4-4 pairs: dist^2 >= 4 (enforced by dot <= 2).
    # - 2-4 pairs: dist^2 >= 4 (enforced by dot <= 1).
    # So min_dist^2 is always >= 4.
    # Max_norm^2 is 4 if we added any 4-norms, else 2.
    # 4 >= 4 is True. 4 >= 2 is True.
    # So it should be valid.
    
    # However, let's double check if any 4-norm vector was actually added.
    # If not, we return just the 220 points.
    
    return final_points.astype(np.int64)

# EVOLVE-BLOCK-END
