import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    D = 11
    
    # We will try to find a large set of integer vectors.
    # The condition is: for all i != j, ||v_i - v_j|| >= max(||v_k|| for all k).
    # This implies that if we fix a target max norm squared R^2, 
    # all vectors must have norm <= R, and pairwise distances >= R.
    # If we use vectors all of norm exactly R, then dist^2 >= R.
    
    # Strategy:
    # 1. Generate a pool of integer vectors with small norms.
    # 2. For each possible max norm squared M (e.g., 2, 3, 4, 5, ...),
    #    try to pack as many vectors of norm <= sqrt(M) as possible
    #    such that pairwise dist^2 >= M.
    #    Note: If a vector has norm < sqrt(M), its distance to others must still be >= sqrt(M).
    #    This is stricter than just using vectors of norm sqrt(M).
    #    
    # Alternative interpretation:
    # The "maximum vector norm" is determined by the set. 
    # If I have a set S, let R = max_{v in S} ||v||.
    # Condition: min_{u!=v in S} ||u-v|| >= R.
    # 
    # If I construct a set where all vectors have norm exactly sqrt(M),
    # then R = sqrt(M), and I need dist^2 >= M.
    # This is a standard spherical code problem on the integer lattice.
    
    best_set = []
    best_score = 0
    
    # Generate candidates for norms squared up to 10
    # Norm squared m = sum of squares of components.
    # Components are integers.
    # To keep the pool manageable, we restrict the number of non-zero components.
    
    # Let's generate all integer vectors with norm squared <= 10.
    # This might be too many if we allow large components, but for small norms, components are small.
    # Max component for norm^2=10 is 3 (3^2=9, remaining 1).
    # So components are in {-3, -2, -1, 0, 1, 2, 3}.
    
    # However, generating all such vectors in 11D is huge: 7^11 ~ 1.9e9. Too many.
    # We need a smarter generation or a smaller norm range.
    
    # Let's stick to norms squared: 2, 3, 4, 5, 6, 8, 9, 10.
    # And restrict support (number of non-zeros) to <= 4 or 5.
    
    # Let's try a greedy approach on a pool of vectors with norm squared = M.
    # We'll try M from 2 to 10.
    
    # Pre-compute all integer vectors with norm squared = M and support <= 4.
    # This is still large. 
    # C(11, 4) * (number of ways to assign signs and magnitudes).
    
    # Let's use a different strategy: 
    # Use the known 224 vectors of norm sqrt(2) as a seed? No, they block each other.
    
    # Let's try M=5. Norm sqrt(5). 
    # Vectors like (1,1,1,1,1,0...) have norm sqrt(5).
    # Or (2,1,0,0...) have norm sqrt(5).
    
    # Let's generate candidates for M=5 with support <= 3.
    # Types for norm^2=5:
    # 1^2 + 1^2 + 1^2 + 1^2 + 1^2 (support 5)
    # 2^2 + 1^2 (support 2)
    # 1^2 + 1^2 + 1^2 + 1^2 (support 4) -- wait, 1+1+1+1=4, not 5.
    # So for M=5:
    # - 2, 1, 0... (support 2)
    # - 1, 1, 1, 1, 1 (support 5)
    
    # Let's generate all integer vectors with norm squared = 5.
    # Support 2: (2, 1) permutations and signs.
    # Support 5: (1, 1, 1, 1, 1) permutations and signs.
    
    candidates = []
    
    # Type 1: Support 2, values {2, 1}
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0]*D
                    v[i] = 2*si
                    v[j] = 1*sj
                    candidates.append(v)
                    # Also swap 2 and 1? No, norm is same, but vector is different.
                    v2 = [0]*D
                    v2[i] = 1*si
                    v2[j] = 2*sj
                    candidates.append(v2)
                    
    # Type 2: Support 5, values {1, 1, 1, 1, 1}
    for combo in itertools.combinations(range(D), 5):
        for signs in itertools.product([1, -1], repeat=5):
            v = [0]*D
            for idx, sign in zip(combo, signs):
                v[idx] = sign
            candidates.append(v)
            
    # Convert to numpy array
    cand_arr = np.array(candidates, dtype=np.int64)
    
    # Shuffle to break symmetry
    np.random.shuffle(cand_arr)
    
    selected = []
    selected_arr = []
    
    # We want to pack as many as possible such that dist^2 >= 5.
    # Since all have norm^2 = 5, R = sqrt(5).
    # Condition: dist^2 >= 5.
    
    # Vectorized greedy:
    # For each candidate, check against all selected.
    # This is O(N * K * D). N ~ 1000, K ~ 200. Should be fast.
    
    for v in cand_arr:
        if not selected_arr:
            selected_arr.append(v)
            continue
            
        # Compute distances to all selected
        # v is shape (D,)
        # selected_arr is list of (D,)
        # Use numpy for speed
        sel = np.array(selected_arr)
        diffs = sel - v
        dists_sq = np.sum(diffs**2, axis=1)
        
        if np.min(dists_sq) >= 5:
            selected_arr.append(v)
            
        if len(selected_arr) > best_score:
            best_score = len(selected_arr)
            best_set = list(selected_arr)
            
        # Check time
        if time.time() > deadline - 10:
            break
            
    # If we found a good set with M=5, return it.
    # The score is len(best_set).
    # Seed score is 224.
    # If best_score > 224, we are improving.
    
    # What if M=5 doesn't yield > 224? 
    # The inspirations suggest 1.27 is possible, which is ~284.
    # M=5 vectors might pack more densely.
    
    # Let's also try M=4 (norm 2).
    # Vectors of norm 2: (2,0...), (1,1,0...).
    # (2,0..) has 22 vectors (11 pos * 2 signs).
    # (1,1,0..) has 224 vectors.
    # Total 246 candidates.
    # Condition: dist^2 >= 4.
    # Let's run a similar greedy for M=4.
    
    candidates_m4 = []
    # Type (2, 0, ...)
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = 2*s
            candidates_m4.append(v)
    # Type (1, 1, 0, ...)
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0]*D
                    v[i] = si
                    v[j] = sj
                    candidates_m4.append(v)
                    
    cand_arr_m4 = np.array(candidates_m4, dtype=np.int64)
    np.random.shuffle(cand_arr_m4)
    
    selected_m4 = []
    
    for v in cand_arr_m4:
        if not selected_m4:
            selected_m4.append(v)
            continue
            
        sel = np.array(selected_m4)
        diffs = sel - v
        dists_sq = np.sum(diffs**2, axis=1)
        
        if np.min(dists_sq) >= 4:
            selected_m4.append(v)
            
        if len(selected_m4) > best_score:
            best_score = len(selected_m4)
            best_set = list(selected_m4)
            
        if time.time() > deadline - 5:
            break
            
    # If no improvement, return the seed 224 vectors (norm sqrt(2))
    if best_score <= 224:
        # Construct the 224 vectors
        base_vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        base_vectors.append(v)
        return np.array(base_vectors, dtype=np.int64)
    else:
        return np.array(best_set, dtype=np.int64)
