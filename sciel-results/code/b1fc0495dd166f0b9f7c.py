import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Generate all candidate vectors with exactly 4 non-zero entries of value +/- 1
    # Norm squared = 4. Norm = 2.
    # Condition: min pairwise distance >= max norm (2).
    # dist^2 >= 4.
    # u, v have norm^2 4.
    # ||u-v||^2 = 4 + 4 - 2 u.v = 8 - 2 u.v >= 4 => u.v <= 2.
    
    candidates = []
    # Iterate over all 4-combinations of indices
    # and all sign combinations
    indices = range(D)
    
    for combo in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(combo, signs):
                v[idx] = sign
            candidates.append(v)
    
    # Convert to numpy array for faster ops? 
    # Or just use lists. 5280 vectors.
    # Greedy selection
    selected = []
    
    # To speed up, we can sort candidates or just iterate.
    # We want a large set.
    # Random shuffling might help find a better packing? 
    # Or just deterministic order.
    # Let's use a randomized greedy approach with multiple restarts if time permits.
    # But 5280 is small. One pass might be enough.
    
    # Optimization: Check condition u.v <= 2.
    
    # We'll do a simple greedy pass.
    # To maximize count, we might want to add vectors that are "far" from others.
    # But standard greedy (first fit) is often decent.
    
    # Let's try to be smart: 
    # If we group by support (the 4 indices), vectors with disjoint supports have u.v = 0 (OK).
    # Vectors with 3 common indices have u.v = +/-1 (OK).
    # Vectors with 2 common indices have u.v = +/-2 (OK if 2, BAD if -2? No, <=2 is OK).
    # Wait, if u.v = -2, then dist^2 = 8 - 2(-2) = 12 >= 4. OK.
    # If u.v = 2, then dist^2 = 8 - 4 = 4 >= 4. OK.
    # If u.v = 3? Not possible with 4 non-zeros?
    # Max dot product:
    # u=(1,1,1,1), v=(1,1,1,1) -> u.v=4. dist^2=0. BAD.
    # u=(1,1,1,1), v=(1,1,1,-1) -> u.v=2. dist^2=4. OK.
    # u=(1,1,1,1), v=(1,1,-1,-1) -> u.v=0. dist^2=8. OK.
    # u=(1,1,1,1), v=(-1,1,1,1) -> u.v=2. OK.
    # u=(1,1,1,1), v=(1,1,1,1) is the only one with u.v=4?
    # Yes, identical vector.
    # What about u=(1,1,1,1) and v=(1,1,1,1) shifted? No, fixed support.
    # So we just need to avoid identical vectors and maybe some close ones?
    # Actually, u.v <= 2 is the condition.
    # u.v = 4 only if u=v.
    # u.v = 3? 
    # u=(1,1,1,1). v=(1,1,1,1) is 4.
    # v=(1,1,1,-1) is 2.
    # v=(1,1,-1,-1) is 0.
    # v=(1,-1,1,1) is 2.
    # v=(1,1,1,1) is the only 4.
    # Is it possible to have u.v=3?
    # Sum of 4 products of +/-1.
    # Possible sums: -4, -2, 0, 2, 4.
    # So u.v is always even.
    # So u.v <= 2 is equivalent to u.v != 4.
    # u.v = 4 implies u and v have same signs on the same support.
    # i.e., u = v.
    # So, **ANY** set of distinct vectors from this candidate list satisfies the condition!
    # Let's verify.
    # If u != v, can u.v = 4?
    # u.v = sum u_i v_i.
    # If supports are different, u.v is sum over intersection.
    # Max intersection size 4.
    # If supports are same, u.v = sum of 4 products.
    # If u != v, at least one sign differs.
    # Then sum is 4-2k where k is number of sign differences.
    # k >= 1.
    # Max sum = 4-2 = 2.
    # If supports are different, intersection size < 4.
    # Max sum = 3 (if 3 common indices and signs match).
    # But sum of 3 products of +/-1 is +/-1 or +/-3.
    # So max is 3.
    # 3 <= 2? No.
    # Wait.
    # If u=(1,1,1,1,0..) and v=(1,1,1,0,1..).
    # Support intersection: 3 indices.
    # u.v = 1*1 + 1*1 + 1*1 = 3.
    # dist^2 = 8 - 2(3) = 2.
    # 2 < 4. Condition FAILS.
    
    # So we CANNOT take all distinct vectors.
    # We need u.v <= 2.
    # This means we must avoid pairs with dot product 3 (and 4).
    # Dot product 3 occurs when supports intersect in 3 indices and signs match on those 3.
    
    # So we need to select a subset such that no two vectors have 3 common indices with matching signs.
    
    # Let's implement the greedy check.
    
    # To make it faster, we can represent vectors as integers (bitmasks for indices and signs).
    # But 5280 is small.
    
    # Greedy:
    # Shuffle candidates randomly?
    # Or just iterate.
    # If we iterate in a structured way, we might pack more.
    
    # Let's try to maximize the set.
    # We can use a simple greedy:
    # selected = []
    # for v in candidates:
    #    ok = True
    #    for u in selected:
    #       if dot(u, v) > 2:
    #          ok = False
    #          break
    #    if ok:
    #       selected.append(v)
    
    # This is O(N^2). 5280^2 = 28M.
    # In Python, this might take ~10-20 seconds.
    # We have 150s.
    # We can do multiple random restarts if we have time.
    
    best_set = []
    
    # Pre-compute dot products? No, too much memory.
    # Just compute on the fly.
    
    # To speed up, we can store selected vectors as tuples of (indices, signs).
    # And compute dot product quickly.
    
    # Let's try one deterministic pass first.
    # Then if time permits, random restarts.
    
    def dot_product(v1, v2):
        # v1, v2 are lists of ints
        return sum(a * b for a, b in zip(v1, v2))
    
    # To optimize dot product, since most are 0, we can store sparse representation.
    # (index, sign) pairs.
    
    sparse_candidates = []
    for v in candidates:
        sparse = [(i, v[i]) for i in range(D) if v[i] != 0]
        sparse_candidates.append(sparse)
    
    def dot_sparse(s1, s2):
        # s1, s2 are lists of (idx, val)
        # Since s1 and s2 are small (len 4), we can just compare.
        # Or convert to dict?
        # 4x4 = 16 comparisons.
        d = 0
        for i, vi in s1:
            for j, vj in s2:
                if i == j:
                    d += vi * vj
                    break
        return d
    
    # Greedy
    selected_sparse = []
    for sp in sparse_candidates:
        ok = True
        for sp_existing in selected_sparse:
            if dot_sparse(sp, sp_existing) > 2:
                ok = False
                break
        if ok:
            selected_sparse.append(sp)
            
    current_count = len(selected_sparse)
    
    # Convert back to dense for return
    def to_dense(sp):
        v = [0] * D
        for i, val in sp:
            v[i] = val
        return v
    
    best_vectors = [to_dense(sp) for sp in selected_sparse]
    
    # Check time
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    # If we have time, try random restarts
    if remaining > 10:
        # We can try to improve by removing one vector and adding two? 
        # Or just random shuffle greedy.
        import random
        best_count = current_count
        best_best = best_vectors
        best_best_sparse = selected_sparse
        
        restarts = 0
        while time.time() - start_time < deadline - 5:
            random.shuffle(sparse_candidates)
            sel = []
            for sp in sparse_candidates:
                ok = True
                for sp_ex in sel:
                    if dot_sparse(sp, sp_ex) > 2:
                        ok = False
                        break
                if ok:
                    sel.append(sp)
            if len(sel) > best_count:
                best_count = len(sel)
                best_best = [to_dense(sp) for sp in sel]
                best_best_sparse = sel
                print(f"New best: {best_count}")
            restarts += 1
            
    return np.array(best_best, dtype=np.int64)

# EVOLVE-BLOCK-END
