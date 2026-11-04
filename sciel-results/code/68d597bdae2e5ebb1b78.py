<<<<<<< SEARCH
def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Returns: {'U', 'V', 'W'}: factor matrices with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 3, 3, 6
    nm, mp, pn = n * m, m * p, p * n  # 9, 18, 18
    
    # Schoolbook baseline: rank 54
    best_rank = n * m * p
    rank = best_rank
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    
    # Strategy: Use structured low-rank decompositions
    # Key idea: C[i,k] = sum_j A[i,j]*B[j,k]
    # We can exploit that A is 3x3 (small) and B is 3x6
    
    # Approach: For each "combination pattern" of A and B indices,
    # try to find rank-1 terms that cover multiple (i,j,k) triples
    
    # Let's try a systematic search with structured initialization
    # Use the fact that entries should be integer/half-integer
    
    import random
    
    def try_structured_rank(r, seed=42, max_attempts=100):
        """Try to find rank-r decomposition with structured search."""
        rng = random.Random(seed)
        
        # Initialize with a pattern-based approach
        # Start from schoolbook and try to merge terms
        
        # Method: Start with rank r, initialize with small integers/half-integers
        # Then do coordinate descent with fast numpy error
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Initialize with random small values
            vals = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
            U_c = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(nm)], dtype=float)
            V_c = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(mp)], dtype=float)
            W_c = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(pn)], dtype=float)
            
            # Coordinate descent: iterate over entries, try small perturbations
            err = _fast_error(U_c.tolist(), V_c.tolist(), W_c.tolist(), n, m, p)
            
            if err < 1e-9:
                return (U_c.tolist(), V_c.tolist(), W_c.tolist(), r)
            
            # Try to improve via coordinate descent
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Pick random coordinate
                mat_choice = rng.choice([0, 1, 2])
                if mat_choice == 0:
                    mat = U_c
                    nrows = nm
                elif mat_choice == 1:
                    mat = V_c
                    nrows = mp
                else:
                    mat = W_c
                    nrows = pn
                
                row = rng.randint(0, nrows - 1)
                col = rng.randint(0, r - 1)
                old_val = mat[row, col]
                
                # Try nearby values
                best_err = err
                best_val = old_val
                for delta in [-1.0, -0.5, 0.5, 1.0]:
                    new_val = old_val + delta
                    mat[row, col] = new_val
                    new_err = _fast_error(
                        U_c.tolist() if mat_choice == 0 else None,
                        V_c.tolist() if mat_choice == 1 else None,
                        W_c.tolist() if mat_choice == 2 else None,
                        n, m, p
                    )
                    # Fix: need to pass correct matrices
                    if mat_choice == 0:
                        new_err = _fast_error(U_c.tolist(), V_c.tolist(), W_c.tolist(), n, m, p)
                    elif mat_choice == 1:
                        new_err = _fast_error(U_c.tolist(), V_c.tolist(), W_c.tolist(), n, m, p)
                    else:
                        new_err = _fast_error(U_c.tolist(), V_c.tolist(), W_c.tolist(), n, m, p)
                    
                    if new_err < best_err:
                        best_err = new_err
                        best_val = new_val
                mat[row, col] = best_val
                err = best_err
                
                if err < 1e-9:
                    return (U_c.tolist(), V_c.tolist(), W_c.tolist(), r)
        
        return None
    
    # Try structured ranks
    # Theoretical: R(3,3,6) is between ~19 and 54
    # Try progressively higher ranks
    ranks_to_try = [27, 30, 32, 36, 38, 40, 42, 45]
    
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
        if r >= best_rank:
            continue
        
        result = try_structured_rank(r, seed=r, max_attempts=30)
        if result is not None:
            U_c, V_c, W_c, r_found = result
            if _verify(U_c, V_c, W_c, n, m, p):
                best_rank = r_found
                best_result = {"U": U_c, "V": V_c, "W": W_c}
                break
    
    return best_result
=======
def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Returns: {'U', 'V', 'W'}: factor matrices with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 3, 3, 6
    nm, mp, pn = n * m, m * p, p * n  # 9, 18, 18
    
    # Schoolbook baseline: rank 54
    best_rank = n * m * p
    rank = best_rank
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    
    import random
    
    def _snap_to_half(x):
        """Snap value to nearest half-integer."""
        return round(x * 2) / 2.0
    
    def _try_merge(U, V, W, idx1, idx2, n, m, p):
        """Try to merge two rank-1 terms into one. Returns (new_U, new_V, new_W, success)."""
        nm, mp, pn = n * m, m * p, p * n
        r = U.shape[1]
        if r < 2:
            return None, None, None, False
        
        # Remove columns idx1 and idx2, add one new column
        # New column should satisfy: sum over d of U_new[a,d] * V_new[b,d] * W_new[c,d] = original
        # Try: new term = weighted combination
        # We need to find U_new, V_new, W_new such that the combined effect matches
        
        # Strategy: try to express the sum of two rank-1 tensors as one
        # This is generally not possible, but we can try various combinations
        
        # Create new matrices with r-1 columns
        keep = [i for i in range(r) if i not in [idx1, idx2]]
        U_new = np.zeros((nm, r - 1))
        V_new = np.zeros((mp, r - 1))
        W_new = np.zeros((pn, r - 1))
        for new_col, old_col in enumerate(keep):
            U_new[:, new_col] = U[:, old_col]
            V_new[:, new_col] = V[:, old_col]
            W_new[:, new_col] = W[:, old_col]
        
        # Now try to add a new column that captures the merged effect
        # The two removed terms contributed: U[:,idx1]*V[:,idx1]*W[:,idx1] + U[:,idx2]*V[:,idx2]*W[:,idx2]
        # We want a single term U_new[:,r-1]*V_new[:,r-1]*W_new[:,r-1] that approximates this
        
        # Try: average the vectors
        U_merged = (U[:, idx1] + U[:, idx2]) / 2.0
        V_merged = (V[:, idx1] + V[:, idx2]) / 2.0
        W_merged = (W[:, idx1] + W[:, idx2]) / 2.0
        
        # Snap to half-integers
        U_merged = np.array([_snap_to_half(x) for x in U_merged])
        V_merged = np.array([_snap_to_half(x) for x in V_merged])
        W_merged = np.array([_snap_to_half(x) for x in W_merged])
        
        # Check if this works
        U_new[:, r - 1] = U_merged
        V_new[:, r - 1] = V_merged
        W_new[:, r - 1] = W_merged
        
        err = _full_error(U_new, V_new, W_new, n, m, p)
        if err < 1e-9:
            return U_new, V_new, W_new, True
        return None, None, None, False
    
    def _local_search(U, V, W, max_iters=500, deadline_check=None):
        """Local search to improve a decomposition."""
        nm, mp, pn = n * m, m * p, p * n
        r = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        err = _fast_error(U, V, W, n, m, p)
        if err < 1e-9:
            return U, V, W, True
        
        rng = np.random.RandomState(42)
        for it in range(max_iters):
            if deadline_check and time.time() > deadline_check:
                break
            
            # Pick random matrix and entry
            mat_choice = rng.randint(0, 3)
            if mat_choice == 0:
                mat, nrows = U, nm
            elif mat_choice == 1:
                mat, nrows = V, mp
            else:
                mat, nrows = W, pn
            
            row = rng.randint(0, nrows)
            col = rng.randint(0, r)
            old_val = mat[row, col]
            
            best_err = err
            best_val = old_val
            for delta in [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]:
                new_val = _snap_to_half(old_val + delta)
                if new_val == old_val:
                    continue
                mat[row, col] = new_val
                new_err = _fast_error(U, V, W, n, m, p)
                if new_err < best_err:
                    best_err = new_err
                    best_val = new_val
            mat[row, col] = best_val
            err = best_err
            
            if err < 1e-9:
                return U, V, W, True
        
        return U, V, W, err < 1e-9
    
    def _greedy_merge(U, V, W, deadline_check=None):
        """Greedy merge: repeatedly try to merge pairs of terms."""
        nm, mp, pn = n * m, m * p, p * n
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        r = U.shape[1]
        merged = True
        while merged:
            if deadline_check and time.time() > deadline_check:
                break
            merged = False
            r = U.shape[1]
            for i in range(r):
                if deadline_check and time.time() > deadline_check:
                    break
                for j in range(i + 1, r):
                    U_new, V_new, W_new, success = _try_merge(U, V, W, i, j, n, m, p)
                    if success:
                        U, V, W = U_new, V_new, W_new
                        merged = True
                        break
                if merged:
                    break
        
        # Local search after merging
        U, V, W, success = _local_search(U, V, W, max_iters=200, deadline_check=deadline_check)
        return U, V, W, U.shape[1]
    
    # Phase 1: Greedy merge from schoolbook
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_merged, V_merged, W_merged, r_merged = _greedy_merge(U_arr, V_arr, W_arr, deadline=deadline - 30)
    
    if r_merged < best_rank:
        if _verify(U_merged, V_merged, W_merged, n, m, p):
            best_rank = r_merged
            best_result = {
                "U": U_merged.tolist(),
                "V": V_merged.tolist(),
                "W": W_merged.tolist()
            }
    
    # Phase 2: Try structured random initializations with local search
    vals = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
    
    for r in range(24, best_rank + 1):
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            continue
        
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.RandomState(r * 100 + attempt)
            U_c = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(nm)], dtype=float)
            V_c = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(mp)], dtype=float)
            W_c = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(pn)], dtype=float)
            
            U_c, V_c, W_c, success = _local_search(U_c, V_c, W_c, max_iters=100, deadline_check=deadline - 5)
            
            if success and _verify(U_c, V_c, W_c, n, m, p):
                best_rank = r
                best_result = {
                    "U": U_c.tolist(),
                    "V": V_c.tolist(),
                    "W": W_c.tolist()
                }
                break
    
    # Phase 3: Final polish with full error check
    if best_rank < n * m * p:
        U_final = np.array(best_result["U"], dtype=float)
        V_final = np.array(best_result["V"], dtype=float)
        W_final = np.array(best_result["W"], dtype=float)
        
        U_final, V_final, W_final, success = _local_search(U_final, V_final, W_final, max_iters=300, deadline_check=deadline - 5)
        
        if _verify(U_final, V_final, W_final, n, m, p):
            best_result = {
                "U": U_final.tolist(),
                "V": V_final.tolist(),
                "W": W_final.tolist()
            }
    
    return best_result
>>>>>>> REPLACE
