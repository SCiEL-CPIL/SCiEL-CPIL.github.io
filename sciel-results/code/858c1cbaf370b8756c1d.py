import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3 # alpha^2 = 3
    
    # Field operations
    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def sub_f(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
        
    def mul_f(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
        
    # Precompute all points
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
                    
    n_points = q * q
    
    # Map point to index
    point_to_idx = {pt: i for i, pt in enumerate(points)}
    
    # Precompute direction indices
    # Directions are lines through origin.
    # There are q+1 = 50 directions.
    # We assign an index 0..49 to each direction.
    # Direction of vector v (v != 0) is the unique line through 0 containing v.
    # We can canonicalize the vector: find the representative with first non-zero component = 1?
    # Or just group them.
    
    # Let's find all 50 direction vectors.
    # A direction is defined by a slope m in F_q or infinity.
    # v = (1, m) or (0, 1).
    
    dir_vectors = []
    for ma in range(p):
        for mb in range(p):
            dir_vectors.append(((1, 0), (ma, mb)))
    dir_vectors.append(((0, 1), (0, 0)))
    
    # Map each non-zero vector to its direction index
    # vec_to_dir = {}
    # for d_idx, dv in enumerate(dir_vectors):
    #     for t_a in range(1, p): # t != 0
    #         for t_b in range(p):
    #             # t = (t_a, t_b) != 0
    #             if t_a == 0 and t_b == 0: continue
    #             t = (t_a, t_b)
    #             v = mul_f(t, dv[0]), mul_f(t, dv[1])
    #             vec_to_dir[v] = d_idx
    #             vec_to_dir[(-v[0][0], -v[0][1]), (-v[1][0], -v[1][1])] = d_idx # -v is same direction?
    #             # In F_7, -1 = 6.
    #             # The line through 0 contains v and -v.
    #             # So yes, same direction.
    
    # Actually, simpler:
    # For each point x, y, compute v = y - x.
    # Find d_idx such that v is a multiple of dir_vectors[d_idx].
    
    # Precompute dir_idx_matrix [n_points][n_points]
    dir_idx_matrix = np.zeros((n_points, n_points), dtype=np.int8)
    
    # Helper to get direction index of a vector v
    # v = (va, vb)
    # If v == (0,0), undefined (but we only use for x != y)
    
    # Build a lookup for vector -> dir_idx
    vec_to_dir = {}
    for d_idx, dv in enumerate(dir_vectors):
        dx, dy = dv
        for t_a in range(p):
            for t_b in range(p):
                if t_a == 0 and t_b == 0:
                    continue
                t = (t_a, t_b)
                tx = mul_f(t, dx)
                ty = mul_f(t, dy)
                v = (tx, ty)
                vec_to_dir[v] = d_idx
                # Add negative?
                # -v is also on the line.
                nv = ((-tx[0]) % p, (-tx[1]) % p), ((-ty[0]) % p, (-ty[1]) % p)
                vec_to_dir[nv] = d_idx
                
    for i in range(n_points):
        pi = points[i]
        for j in range(n_points):
            if i == j:
                continue
            pj = points[j]
            v = sub_f(pj[0], pi[0]), sub_f(pj[1], pi[1])
            # v is vector from i to j
            dir_idx_matrix[i][j] = vec_to_dir[v]
            
    # Greedy Algorithm
    # bad_mask[y] is a bitmask of directions that are "bad" for y
    # i.e., there exists x in S such that x is on line (y, m)
    bad_mask = [0] * n_points
    in_S = [False] * n_points
    candidates = list(range(n_points))
    
    # We want to maximize |S|
    # Greedy: pick x with minimum impact
    # Impact(x) = number of y != x such that bit dir_idx_matrix[y][x] is NOT in bad_mask[y]
    
    S = []
    
    while candidates:
        best_x = -1
        min_impact = n_points + 1
        best_valid = False
        
        # Calculate impact for all candidates
        # To speed up, we can break early if we find a valid one with low impact?
        # No, we need the min impact among valid ones.
        
        # We can precompute the "new bits" for each candidate?
        
        for x in candidates:
            impact = 0
            valid = True
            
            for y in range(n_points):
                if y == x:
                    continue
                d = dir_idx_matrix[y][x]
                bit = 1 << d
                if (bad_mask[y] & bit) == 0:
                    impact += 1
                    # Check if this would make bad_count > 49
                    # If bad_mask[y] already has 49 bits set, adding one more makes 50.
                    # popcount check
                    # We can maintain popcount array
                    pass
                    
            # We need popcount. Let's maintain it.
            # This loop is O(N^2) inside the while loop.
            # Total O(N^3) ? No, candidates shrinks.
            # Sum of |candidates| over iterations is O(N^2).
            # Inside, we loop over N points.
            # Total O(N^3) = 2401^3 ~ 1.3e10. Too slow.
            
            # Optimization:
            # We only care about y where the bit is NOT set.
            # And we only care if popcount(bad_mask[y]) == 49.
            
            # Let's maintain popcount array.
            pass
            
        # This approach is too slow in pure Python if N=2401.
        # We need vectorization or a smarter heuristic.
        
        # Heuristic: Just pick a random candidate that is valid?
        # Or pick the one with min degree in the "conflict graph"?
        
        # Let's try a simpler greedy:
        # Pick x randomly from candidates.
        # If valid, add to S.
        # Else, remove from candidates.
        
        # To make it better, we can prioritize points that are "isolated" in the current S?
        
        # Let's stick to the current program's structure but improve the heuristic.
        # The current program uses random order.
        # Let's use "Min-Conflict" style:
        # For each x in candidates, count how many y would become invalid.
        # If count > 0, x is invalid.
        # Among valid x, pick one with min "impact" (number of y whose bad_count increases).
        
        # To speed up, we can approximate impact.
        
        # Given the time constraint, I will implement a vectorized version of the "check all" logic using NumPy.
        
        # bad_mask is an array of uint64.
        # dir_idx_matrix is an array of uint8.
        
        # For a given x, we want to check all y.
        # bits = 1 << dir_idx_matrix[:, x]
        # new_mask = bad_mask | bits
        # new_popcount = np.bit_count(new_mask) # NumPy >= 1.24?
        # If not, use lookup table.
        
        # Let's use a lookup table for popcount of 16-bit chunks or just use bin(x).count('1') if slow?
        # bin(x).count('1') is fast enough for 2401 items?
        
        # Let's try the vectorized check for a single x.
        
        pass
        
    # Fallback: Use the current program's logic but with a better ordering.
    # The current program is likely O(N^3) or O(N^2 * D).
    
    # I will provide a solution that uses the vectorized bitmask approach, which is O(N^2) per restart if optimized, or O(N^2) total if we just pick valid ones greedily without full impact calculation (just validity).
    
    # Revised Plan:
    # 1. Precompute dir_idx_matrix.
    # 2. Initialize bad_mask, popcount.
    # 3. candidates = all points.
    # 4. While candidates:
    #    - Randomly shuffle candidates? No, deterministic is better for reproducibility, but random helps escape local minima.
    #    - Pick x from candidates (e.g., first one, or min impact).
    #    - Check validity of x using vectorized ops.
    #    - If valid:
    #       - Update bad_mask and popcount for all y using vectorized ops.
    #       - Add x to S.
    #       - Remove x from candidates.
    #    - Else:
    #       - Remove x from candidates.
       
    # This is O(N^2) vectorized ops.
    # N=2401. N^2 = 5.7e6.
    # Each op is on arrays of size N.
    # 5.7e6 vector ops of size 2401? No.
    # In each iteration, we process one x.
    # We do vectorized ops on arrays of size N.
    # Total iterations N.
    # Total work N * (vectorized ops on N) = N^2 scalar ops effectively.
    # This should be fast.

    bad_mask = np.zeros(n_points, dtype=np.uint64)
    popcount = np.zeros(n_points, dtype=np.int8)
    
    # Precompute 1 << d for d in 0..49
    bits = [1 << d for d in range(50)]
    
    S_indices = []
    candidates = list(range(n_points))
    
    # To improve quality, we can do multiple restarts with different random shuffles of candidates.
    
    best_S_size = 0
    best_S = []
    
    for restart in range(20):
        candidates = list(range(n_points))
        np.random.shuffle(candidates)
        
        bad_mask = np.zeros(n_points, dtype=np.uint64)
        popcount = np.zeros(n_points, dtype=np.int8)
        S_indices = []
        
        # We iterate through candidates.
        # We need to check if x is valid.
        # Valid if for all y, popcount(bad_mask[y] | (1 << dir_idx_matrix[y][x])) <= 49.
        
        # We can vectorize the check for a single x.
        
        # dir_col = dir_idx_matrix[:, x] # shape (N,)
        # bit_col = bits[dir_col] # shape (N,)
        # new_mask = bad_mask | bit_col
        # new_pop = popcount(new_mask)
        # valid = np.all(new_pop <= 49)
        
        # If valid, update:
        # bad_mask = new_mask
        # popcount = new_pop
        # S_indices.append(x)
        
        # Else, skip x.
        
        # Note: dir_idx_matrix[i][i] is 0, but we ignore y=x.
        # For y=x, the direction is undefined, but we don't care about y=x.
        # We should set dir_idx_matrix[x][x] to a dummy value that doesn't affect the check, or mask it out.
        # Let's mask out y=x by setting bit to 0? No, 1<<0 is 1.
        # We can just check `new_pop <= 49` for all y, and for y=x, bad_mask[x] doesn't change?
        # Wait, if y=x, we are adding x to S.
        # Does x affect x?
        # Bad(x) = { m | exists y in S, y!=x, y in L(x,m) }.
        # x is not in S \ {x}. So x does not add to Bad(x).
        # So for y=x, the bit should NOT be set.
        # So we should ensure that for y=x, the bit is 0.
        
        # Let's create a masked bit array.
        
        for x in candidates:
            # Get direction indices for all y pointing to x
            d_col = dir_idx_matrix[:, x]
            
            # Create bit array
            # bits is a list, we need to map d_col to bits
            # We can precompute a lookup table for bits?
            # Or use np.take(bits_array, d_col)
            
            # bits_array = np.array([1 << d for d in range(50)], dtype=np.uint64)
            
            bit_col = bits_array[d_col]
            
            # Mask out y=x
            bit_col[x] = 0
            
            new_mask = bad_mask | bit_col
            new_pop = popcount_arr[new_mask] # Lookup table for popcount
            
            # Check if any new_pop > 49
            if np.any(new_pop > 49):
                continue # Invalid, skip x
            
            # Valid, update
            bad_mask = new_mask
            popcount = new_pop
            S_indices.append(x)
            
        if len(S_indices) > best_S_size:
            best_S_size = len(S_indices)
            best_S = S_indices.copy()
            
    # Construct N
    N = [points[i] for i in range(n_points) if i not in set(best_S)]
    
    arr = np.array(N, dtype=np.int64)
    return arr
