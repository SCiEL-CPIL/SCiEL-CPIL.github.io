import numpy as np
import time
import random
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Strategy:
    # 1. Generate a diverse pool of integer vectors with small norms (1, 2, 3, 4).
    #    - Norm^2 = 1: weight 1 (11 vectors)
    #    - Norm^2 = 2: weight 2 (220 vectors)
    #    - Norm^2 = 3: weight 3 (1650 vectors)
    #    - Norm^2 = 4: weight 4 (5280 vectors)
    #    Total pool size ~7200.
    #
    # 2. The constraint is: for all i != j, ||v_i - v_j|| >= max_k ||v_k||.
    #    Let R^2 = max_k ||v_k||^2.
    #    We need ||v_i - v_j||^2 >= R^2.
    #    ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i . v_j.
    #
    # 3. Greedy Construction:
    #    - Start with an empty set.
    #    - Maintain the current maximum squared norm R^2 in the set.
    #    - At each step, find a vector v from the pool that is compatible with all current vectors.
    #    - Compatibility: ||v - v_i||^2 >= R^2 for all v_i in current set.
    #    - To allow adding vectors with higher norms later, we might want to prioritize lower norm vectors first?
    #      Actually, if we add a high norm vector, R increases, making it harder to add more.
    #      So, a good heuristic is to sort the pool by norm (ascending) and greedily add if compatible.
    #      However, this might get stuck.
    #      Better: Randomized Greedy with Restart.
    #      In each trial, shuffle the pool. Iterate through. Add vector if compatible with current set.
    #      Track the best set size found.
    #
    # 4. Optimization:
    #    - Precompute norms squared for the pool.
    #    - Use vectorized operations to check compatibility with the current set.
    #    - Current set is small (max ~500), pool is ~7000.
    #    - Checking compatibility of one candidate v against current set S:
    #      dists = ||v - S||^2. We need min(dists) >= R^2.
    #      dists = ||v||^2 + ||S||^2 - 2 v . S.
    #      This can be computed quickly using dot products.
    
    # Generate Pool
    pool = []
    norms_sq = []
    
    # Weight 1 (norm^2=1)
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            pool.append(v)
            norms_sq.append(1)
            
    # Weight 2 (norm^2=2)
    for comb in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(comb, signs):
                v[pos] = s
            pool.append(v)
            norms_sq.append(2)
            
    # Weight 3 (norm^2=3)
    for comb in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(comb, signs):
                v[pos] = s
            pool.append(v)
            norms_sq.append(3)
            
    # Weight 4 (norm^2=4)
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(comb, signs):
                v[pos] = s
            pool.append(v)
            norms_sq.append(4)
            
    pool_arr = np.array(pool, dtype=np.int64)
    norms_arr = np.array(norms_sq, dtype=np.int64)
    n_pool = len(pool_arr)
    
    best_set_indices = []
    best_size = 0
    
    # We will run multiple randomized greedy trials.
    # To speed up, we can pre-compute the Gram matrix of the pool? 
    # n_pool is ~7200. 7200x7200 matrix is ~50M entries. 
    # int8 or int16 might be enough? Dots are small.
    # But computing it takes time and memory. 
    # Let's try to compute on the fly or in blocks if needed.
    # Actually, for greedy, we check candidate v against current set S.
    # If S is stored as indices, we can compute dots v . pool[S].
    
    # Let's optimize the inner loop.
    # Instead of checking every candidate against every member of S in Python, 
    # we can vectorize: 
    # For a candidate v, dists_sq = norms_arr[S] + norm_sq_v - 2 * (pool_arr[S] @ v)
    # We need np.min(dists_sq) >= R_sq.
    
    # We will maintain:
    # current_indices: list of indices in pool_arr
    # current_norms_sq: norms of vectors in current_indices
    # R_sq: max norm sq in current set
    
    # To make it faster, we can sort the pool by norm? 
    # The cell finding suggests "norm-descending preference" was used in a successful variant, 
    # but usually starting small allows more vectors. 
    # However, if we start with large norms, R is large immediately, blocking many.
    # If we start with small norms, R is small, allowing more, but then we might add a large one later 
    # which increases R and might invalidate previous assumptions? No, the constraint is global.
    # Adding a vector v with norm > R increases R to ||v||^2. 
    # Then we must re-check all existing vectors u against the new R.
    # ||u - v||^2 >= new_R?
    # This is expensive to re-check.
    
    # Alternative: Fix the target R. 
    # If we decide the final set will have max norm R, we only pick vectors with norm <= R,
    # and ensure all pairwise distances >= R.
    # We can try R=1, 2, 3, 4.
    # If we pick R=4, we can only pick vectors with norm <= 4 (all of them), but dist >= 4.
    # If we pick R=2, we can only pick vectors with norm <= 2, and dist >= 2.
    # The "220" solution was R=2, using only weight-2 vectors (norm^2=2).
    # Wait, if R=2, we can use weight-1 (norm^2=1) and weight-2 (norm^2=2) vectors.
    # Distance between two weight-1 vectors (orthogonal): ||e_i - e_j||^2 = 1+1-0=2 >= 2. OK.
    # Distance between weight-1 and weight-2: ||e_i - (e_i+e_j)||^2 = 1+2-2=1 < 2. FAIL.
    # So mixing norms is tricky.
    
    # Let's stick to the randomized greedy on the full pool, but handle the R update carefully.
    # To avoid re-checking, we can just enforce the constraint at the time of addition:
    # When adding v, let R_new = max(R_old, ||v||^2).
    # We need ||v - u||^2 >= R_new for all u in S.
    # And for all u, w in S, we already have ||u - w||^2 >= R_old.
    # If R_new > R_old, we need to check if ||u - w||^2 >= R_new for all u,w in S.
    # This is O(|S|^2). If |S| is ~500, this is 250k ops. Too slow per addition.
    
    # Heuristic: Only allow adding vectors that do NOT increase R.
    # i.e., sort pool by norm descending? No, ascending.
    # If we process in ascending norm, R is non-decreasing.
    # If we add v with norm^2 = R_old, R doesn't change. Easy check.
    # If we add v with norm^2 > R_old, R increases. 
    # To avoid the re-check, we can restrict our search to vectors of a SINGLE norm R.
    # This brings us back to the previous strategy.
    
    # How to beat 220?
    # The cell finding says "Expands candidate pool... using greedy selection with norm-descending preference... 1.991".
    # This implies mixing norms IS possible and beneficial.
    # How did they handle the R increase?
    # Maybe they just checked against the CURRENT R, and if R increased, they accepted the risk? 
    # No, the evaluator checks the final set.
    # If we add v and R increases, we MUST verify the whole set.
    # Perhaps the set size is small enough that O(|S|^2) is acceptable for a few additions?
    # Or, they only add vectors that keep R constant?
    
    # Let's try a hybrid:
    # 1. Start with a set of weight-4 vectors (R=4). Find a large independent set.
    # 2. Try to add weight-3, 2, 1 vectors that are compatible with R=4.
    #    For a vector v with norm^2 < 4, we need ||v - u||^2 >= 4 for all u in S (where u has norm^2=4).
    #    ||v - u||^2 = ||v||^2 + 4 - 2 v.u >= 4  => ||v||^2 >= 2 v.u.
    #    This is a strict constraint.
    
    # Let's try the opposite:
    # 1. Start with weight-1 vectors (R=1).
    #    Max set of weight-1 vectors: 11 (e_i) or 22 (e_i, -e_i).
    #    Dist(e_i, -e_j) = 2 if i!=j, 0 if i=j (not allowed).
    #    Dist(e_i, e_j) = 2 if i!=j.
    #    So all 22 weight-1 vectors work? 
    #    ||e_i - (-e_i)|| = 2 >= 1. Yes.
    #    So 22 is a base.
    # 2. Add weight-2 vectors. R becomes 2.
    #    Need ||v - u||^2 >= 2 for all u in S (weight-1 and weight-2).
    #    This might allow growing the set.
    
    # Let's implement a greedy that allows R to grow, but performs the full verification only when R grows.
    # Since R can only grow 1->2->3->4, we will have at most 3 "expensive" full verifications.
    # The additions at a fixed R are cheap (O(|S|) per candidate).
    
    def verify_set(indices, R_sq):
        # Check if all pairwise distances in indices are >= R_sq
        if len(indices) < 2:
            return True
        vecs = pool_arr[indices]
        # Compute pairwise distances
        # dist^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j
        # Use broadcasting
        dots = vecs @ vecs.T
        norms_sq_vec = norms_arr[indices]
        dists_sq = norms_sq_vec[:, None] + norms_sq_vec[None, :] - 2 * dots
        # Check min off-diagonal
        np.fill_diagonal(dists_sq, np.inf)
        return np.min(dists_sq) >= R_sq

    def greedy_trial(order):
        current_indices = []
        current_R_sq = 0
        last_R_sq = 0
        
        for idx in order:
            v = pool_arr[idx]
            v_norm_sq = norms_arr[idx]
            new_R_sq = max(current_R_sq, v_norm_sq)
            
            # Check compatibility with current set under new_R_sq
            if not current_indices:
                current_indices.append(idx)
                current_R_sq = new_R_sq
                last_R_sq = new_R_sq
                continue
            
            # Check dist to all in current_indices
            # dist^2 = v_norm_sq + u_norm_sq - 2 v.u
            # We need dist^2 >= new_R_sq
            # If new_R_sq == current_R_sq, we just check against current set.
            # If new_R_sq > current_R_sq, we check against current set AND we know we'll need to re-verify later if accepted.
            
            # Vectorized check
            u_vecs = pool_arr[current_indices]
            u_norms = norms_arr[current_indices]
            dots = u_vecs @ v
            dists_sq = v_norm_sq + u_norms - 2 * dots
            
            if np.min(dists_sq) < new_R_sq:
                continue
            
            # If R increased, we need to verify the whole set against the new R
            # But wait, if we add v, the set becomes current_indices + [idx].
            # We already checked v against current_indices.
            # We need to check current_indices against each other with new_R_sq.
            # If new_R_sq == last_R_sq, no need.
            # If new_R_sq > last_R_sq, we must verify the old set.
            
            if new_R_sq > last_R_sq:
                # Verify old set with new_R_sq
                if not verify_set(current_indices, new_R_sq):
                    continue
            
            # Accept
            current_indices.append(idx)
            current_R_sq = new_R_sq
            last_R_sq = new_R_sq
            
        return current_indices

    # Run multiple trials with different orderings
    # Ordering 1: Ascending norm (start small)
    # Ordering 2: Descending norm (start large)
    # Ordering 3: Random
    
    best_indices = []
    
    # Trial 1: Ascending norm
    indices_sorted_asc = np.argsort(norms_arr)
    # Add some randomness to ties
    random.seed(42)
    indices_shuffled_asc = indices_sorted_asc.copy()
    # Group by norm and shuffle within groups?
    # Simpler: just sort, then perturb.
    for i in range(len(indices_shuffled_asc) - 1, 0, -1):
        j = random.randint(0, i)
        indices_shuffled_asc[i], indices_shuffled_asc[j] = indices_shuffled_asc[j], indices_shuffled_asc[i]
        # This is a full shuffle. Let's do a stable sort then shuffle.
        
    # Actually, let's just do a few specific strategies.
    
    # Strategy A: Ascending norm, greedy
    idx_asc = np.argsort(norms_arr, kind='mergesort')
    res = greedy_trial(idx_asc)
    if len(res) > len(best_indices):
        best_indices = res
    
    # Strategy B: Descending norm, greedy
    idx_desc = np.argsort(-norms_arr, kind='mergesort')
    res = greedy_trial(idx_desc)
    if len(res) > len(best_indices):
        best_indices = res
        
    # Strategy C: Randomized Greedy with Restart
    # We will run several random trials.
    # To save time, we limit the number of trials based on time.
    
    trials_done = 0
    max_trials = 20 # Adjust based on time
    
    while time.time() < deadline - 10:
        # Random order
        idx_rand = np.random.permutation(n_pool)
        res = greedy_trial(idx_rand)
        if len(res) > len(best_indices):
            best_indices = res
            # If we found a large set, maybe do some local search?
            # For now, just record.
            
        trials_done += 1
        if trials_done % 5 == 0:
            # Check time
            if time.time() > deadline - 10:
                break
    
    # If best_indices is empty, fallback to weight-2 set
    if not best_indices:
        # Return all weight-2 vectors
        pool2 = []
        for comb in itertools.combinations(range(D), 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = [0] * D
                for idx2, sign in zip(comb, signs):
                    v[idx2] = sign
                pool2.append(v)
        return np.array(pool2, dtype=np.int64)
        
    return pool_arr[best_indices]

