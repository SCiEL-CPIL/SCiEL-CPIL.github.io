<<<<<<< SEARCH
def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Returns: {'U', 'V', 'W'}: factor matrices with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 6
    nm, mp, pn = n * m, m * p, p * n  # 9, 18, 18
    
    best_rank = n * m * p  # Schoolbook: 54
    best_result = None
    
    # Start with schoolbook as baseline
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    if _verify(U, V, W, n, m, p):
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    import random
    
    # Precompute on-diagonal indices for fast error computation
    on_diag_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                on_diag_indices.append((i * m + j, j * p + k, k * n + i))
    
    # Precompute off-diagonal indices (all except on-diag)
    on_diag_set = set(on_diag_indices)
    off_diag_indices = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in on_diag_set:
                    off_diag_indices.append((a, b, c))
    
    # Precompute as arrays for vectorized operations
    on_diag_a = np.array([x[0] for x in on_diag_indices])
    on_diag_b = np.array([x[1] for x in on_diag_indices])
    on_diag_c = np.array([x[2] for x in on_diag_indices])
    
    off_diag_a = np.array([x[0] for x in off_diag_indices])
    off_diag_b = np.array([x[1] for x in off_diag_indices])
    off_diag_c = np.array([x[2] for x in off_diag_indices])
    
    def compute_error_vec(U_arr, V_arr, W_arr):
        """Vectorized error computation using numpy."""
        # On-diagonal: sum_d U[a,d]*V[b,d]*W[c,d] should be 1
        # Use einsum for batch computation
        U_od = U_arr[on_diag_a, :]  # shape (54, r)
        V_od = V_arr[on_diag_b, :]
        W_od = W_arr[on_diag_c, :]
        on_vals = np.sum(U_od * V_od * W_od, axis=1)
        on_err = np.sum((on_vals - 1.0) ** 2)
        
        # Off-diagonal: should be 0
        # Sample if too many
        if len(off_diag_a) > 500:
            idx = np.random.choice(len(off_diag_a), 500, replace=False)
            od_a = off_diag_a[idx]
            od_b = off_diag_b[idx]
            od_c = off_diag_c[idx]
        else:
            od_a = off_diag_a
            od_b = off_diag_b
            od_c = off_diag_c
        
        U_od2 = U_arr[od_a, :]
        V_od2 = V_arr[od_b, :]
        W_od2 = W_arr[od_c, :]
        off_vals = np.sum(U_od2 * V_od2 * W_od2, axis=1)
        off_err = np.sum(off_vals ** 2)
        
        return on_err + off_err
    
    def try_rank(r, max_iter=100, seed=42, n_attempts=20):
        """Try to find a rank-r decomposition using vectorized random search and refinement."""
        rng = random.Random(seed)
        
        def random_entry():
            return rng.choice([-2, -1, -0.5, 0, 0.5, 1, 2])
        
        for attempt in range(n_attempts):
            if time.time() > deadline:
                return None
            
            U_cand = np.array([[random_entry() for _ in range(r)] for _ in range(nm)])
            V_cand = np.array([[random_entry() for _ in range(r)] for _ in range(mp)])
            W_cand = np.array([[random_entry() for _ in range(r)] for _ in range(pn)])
            
            err0 = compute_error_vec(U_cand, V_cand, W_cand)
            if err0 < 1e-9:
                return (U_cand.tolist(), V_cand.tolist(), W_cand.tolist(), r)
            
            # Local search: perturb one entry at a time
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for iter_i in range(max_iter):
                    if time.time() > deadline:
                        break
                    
                    mat_idx = rng.choice([0, 1, 2])
                    if mat_idx == 0:
                        mat = U_cand
                        nrows = nm
                    elif mat_idx == 1:
                        mat = V_cand
                        nrows = mp
                    else:
                        mat = W_cand
                        nrows = pn
                    
                    row = rng.randint(0, nrows - 1)
                    col = rng.randint(0, r - 1)
                    old_val = mat[row, col]
                    
                    # Try a few neighbor values
                    candidates = [old_val + d for d in [-2, -1, -0.5, 0.5, 1, 2]]
                    candidates += [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]
                    
                    best_err = err0
                    best_val = old_val
                    for cand in candidates:
                        mat[row, col] = cand
                        err = compute_error_vec(U_cand, V_cand, W_cand)
                        if err < best_err:
                            best_err = err
                            best_val = cand
                    
                    mat[row, col] = best_val
                    if best_err < err0:
                        err0 = best_err
                        improved = True
                        if err0 < 1e-9:
                            return (U_cand.tolist(), V_cand.tolist(), W_cand.tolist(), r)
            
            if err0 < 1e-9:
                return (U_cand.tolist(), V_cand.tolist(), W_cand.tolist(), r)
        
        return None
    
    # Try structured approach first: split into two 3x3x3 problems
    # C = A * B where B = [B1 | B2], each 3x3
    # If we can find rank-r1 for 3x3x3, total rank is 2*r1
    # But we can share U factors: use same U for both halves
    # Actually the U factor depends on (i,j) which is the same for both
    # So we can share U columns!
    
    # Let's try: for each half, we need U[i*m+j, d] * V1[j*p+k, d] * W1[k*n+i, d] for first half
    # and U[i*m+j, d'] * V2[j*p+3+k, d'] * W2[(3+k)*n+i, d'] for second half
    # If we use the same U for both: rank = r1 + r1 but U is shared
    # Actually no, U is the same matrix, so we can have:
    # First r1 columns: handle B1
    # Next r1 columns: handle B2
    # Total rank = 2 * r1
    
    # For 3x3x3, best known is 19, so total would be 38
    # But maybe we can do better with a joint decomposition
    
    # Try ranks
    ranks_to_try = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
    
    for r in sorted(ranks_to_try):
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            continue
        
        print(f"Trying rank {r}...", file=sys.stderr)
        result = try_rank(r, max_iter=50, seed=r, n_attempts=15)
        if result is not None:
            U_cand, V_cand, W_cand, r_found = result
            if _verify(U_cand, V_cand, W_cand, n, m, p):
                print(f"Found valid rank {r_found}!", file=sys.stderr)
                best_rank = r_found
                best_result = {"U": U_cand, "V": V_cand, "W": W_cand}
                break
        else:
            print(f"Rank {r}: no solution found in time", file=sys.stderr)
    
    # If we didn't find anything better, return the schoolbook
    if best_result is None:
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
=======
def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Uses Kronecker product structure: the 3x3x6 multiplication tensor is
    the Kronecker product of a 3x3x3 multiplication tensor and a 3x1x2
    broadcast tensor. This allows constructing solutions from 3x3x3
    decompositions.
    
    Returns: {'U', 'V', 'W'}: factor matrices with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 6
    nm, mp, pn = n * m, m * p, p * n  # 9, 18, 18
    
    best_rank = n * m * p  # Schoolbook: 54
    best_result = None
    
    # Start with schoolbook as baseline
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    if _verify(U, V, W, n, m, p):
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    def kron_decompose(U3, V3, W3, r3):
        """Construct 3x3x6 decomposition from a 3x3x3 decomposition using Kronecker product.
        
        The 3x3x6 tensor T6[i,j,k] = 1 if k < 3, else 0 (for the broadcast part)
        combined with T3[i,j,k'] for the 3x3x3 part.
        
        Actually, let me think more carefully.
        
        For 3x3x6: A is 3x3, B is 3x6, C is 3x6.
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        We can write B as [B1 | B2] where B1, B2 are 3x3.
        Then C = [C1 | C2] where C1 = A*B1, C2 = A*B2.
        
        The tensor is T6[a,b,c] where:
        a = i*m+j (i in 0..2, j in 0..2) -> 9 values
        b = j*p+k (j in 0..2, k in 0..5) -> 18 values  
        c = k*n+i (k in 0..5, i in 0..2) -> 18 values
        
        For the Kronecker structure, if we have a 3x3x3 decomposition with rank r3:
        T3[a3, b3, c3] = sum_d U3[a3,d] * V3[b3,d] * W3[c3,d]
        
        where a3 = i*3+j, b3 = j*3+k3, c3 = k3*3+i for k3 in 0..2
        
        For the 3x3x6 case, we need to handle k in 0..5.
        
        Key insight: T6[i,j,k] = 1 for all valid (i,j,k).
        This is NOT a Kronecker product in the simple sense.
        
        However, we can use the structure: for each k in 0..5, the "slice"
        is the same 3x3 multiplication pattern. So we can share the U factor
        and use separate V and W factors for each k.
        
        Actually, a better approach: use the Kronecker product with a 
        1x3x2 broadcast tensor (or similar).
        
        Let me think differently. The 3x3x6 multiplication tensor can be
        viewed as: for each of the 6 columns of B, we do a 3x3x3 multiplication.
        These are independent, so the rank is at most 6 * (rank of 3x3x3).
        
        But we can do better by sharing structure. Let's use the fact that
        the U factor only depends on (i,j), not on k. So we can have:
        
        For rank r = r3 * 2 (splitting B into two 3x3 blocks):
        Columns 0..r3-1: handle B1 (columns 0-2 of B)
        Columns r3..2*r3-1: handle B2 (columns 3-5 of B)
        
        U is the same for both (since it depends only on i,j)
        V has two blocks: V1 for B1, V2 for B2
        W has two blocks: W1 for C1, W2 for C2
        
        This gives rank 2*r3. For 3x3x3 with rank 19, total rank = 38.
        
        But can we do better? What if we interleave?
        
        Actually, the key structural insight is:
        T6[i,j,k] = 1 for all i,j,k in valid range.
        
        This means the tensor is "product-like" in a specific way.
        
        Let me try a direct construction:
        For the 3x3x6 case, we need:
        sum_d U[i*3+j, d] * V[j*6+k, d] * W[k*3+i, d] = 1
        
        One approach: use the 3x3x3 decomposition and replicate.
        """
        # Construct from 3x3x3 decomposition
        # U3: shape (9, r3) for indices a3 = i*3+j
        # V3: shape (9, r3) for indices b3 = j*3+k3 (k3 in 0..2)
        # W3: shape (9, r3) for indices c3 = k3*3+i
        
        # For 3x3x6:
        # U: shape (9, r) where r = 2*r3
        # V: shape (18, r) where b = j*6+k, k in 0..5
        # W: shape (18, r) where c = k*3+i, k in 0..5
        
        U = np.zeros((9, 2 * r3))
        V = np.zeros((18, 2 * r3))
        W = np.zeros((18, 2 * r3))
        
        # First block: k in 0..2 (use V3 and W3 directly)
        for i in range(3):
            for j in range(3):
                a = i * 3 + j
                U[a, :r3] = U3[a, :]
        
        for j in range(3):
            for k3 in range(3):
                b = j * 6 + k3
                V[b, :r3] = V3[j * 3 + k3, :]
        
        for k3 in range(3):
            for i in range(3):
                c = k3 * 3 + i
                W[c, :r3] = W3[k3 * 3 + i, :]
        
        # Second block: k in 3..5 (shift indices)
        for i in range(3):
            for j in range(3):
                a = i * 3 + j
                U[a, r3:] = U3[a, :]
        
        for j in range(3):
            for k3 in range(3):
                b = j * 6 + (k3 + 3)
                V[b, r3:] = V3[j * 3 + k3, :]
        
        for k3 in range(3):
            for i in range(3):
                c = (k3 + 3) * 3 + i
                W[c, r3:] = W3[k3 * 3 + i, :]
        
        return U.tolist(), V.tolist(), W.tolist(), 2 * r3
    
    # Known 3x3x3 decompositions
    # Schoolbook 3x3x3: rank 9
    U3_school = np.zeros((9, 9))
    V3_school = np.zeros((9, 9))
    W3_school = np.zeros((9, 9))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3_school[i * 3 + j, col] = 1
                V3_school[j * 3 + k, col] = 1
                W3_school[k * 3 + i, col] = 1
                col += 1
    
    # Construct 3x3x6 from schoolbook 3x3x3: rank 18
    U18, V18, W18, r18 = kron_decompose(U3_school, V3_school, W3_school, 9)
    if _verify(U18, V18, W18, n, m, p):
        if r18 < best_rank:
            best_rank = r18
            best_result = {"U": U18, "V": V18, "W": W18}
        print(f"Schoolbook Kronecker: rank {r18}", file=sys.stderr)
    
    # Try to find better 3x3x3 decompositions and use Kronecker
    import random
    
    # Precompute for 3x3x3 verification
    def verify_3x3x3(U3, V3, W3):
        U3 = np.array(U3, dtype=float)
        V3 = np.array(V3, dtype=float)
        W3 = np.array(W3, dtype=float)
        r3 = U3.shape[1]
        T3 = np.einsum('ad,bd,cd->abc', U3, V3, W3)
        on_diag = np.zeros((9, 9, 9), dtype=bool)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    on_diag[i*3+j, j*3+k, k*3+i] = True
        if not np.all(np.abs(T3[on_diag] - 1.0) < 1e-9):
            return False
        off = ~on_diag
        if np.any(np.abs(T3[off]) > 1e-9):
            return False
        return True
    
    def search_3x3x3(r3, max_iter=50, n_attempts=10, seed=42):
        """Search for a rank-r3 3x3x3 decomposition."""
        rng = random.Random(seed)
        
        def random_entry():
            return rng.choice([-2, -1, -0.5, 0, 0.5, 1, 2])
        
        for attempt in range(n_attempts):
            if time.time() > deadline:
                return None
            
            U3 = np.array([[random_entry() for _ in range(r3)] for _ in range(9)])
            V3 = np.array([[random_entry() for _ in range(r3)] for _ in range(9)])
            W3 = np.array([[random_entry() for _ in range(r3)] for _ in range(9)])
            
            if verify_3x3x3(U3, V3, W3):
                return U3.tolist(), V3.tolist(), W3.tolist(), r3
            
            # Local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for _ in range(max_iter):
                    if time.time() > deadline:
                        break
                    
                    mat_idx = rng.choice([0, 1, 2])
                    if mat_idx == 0:
                        mat = U3
                    elif mat_idx == 1:
                        mat = V3
                    else:
                        mat = W3
                    
                    row = rng.randint(0, 8)
                    col = rng.randint(0, r3 - 1)
                    old_val = mat[row, col]
                    
                    candidates = [old_val + d for d in [-2, -1, -0.5, 0.5, 1, 2]]
                    candidates += [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]
                    
                    best_val = old_val
                    for cand in candidates:
                        mat[row, col] = cand
                        if verify_3x3x3(U3, V3, W3):
                            return U3.tolist(), V3.tolist(), W3.tolist(), r3
                    
                    mat[row, col] = old_val
                    improved = False  # Simplified: just check if we found a solution
        
        return None
    
    # Try various ranks for 3x3x3, then use Kronecker to get 3x3x6
    ranks_3x3x3 = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    
    for r3 in sorted(ranks_3x3x3):
        if time.time() > deadline - 20:
            break
        
        r6 = 2 * r3
        if r6 >= best_rank:
            continue
        
        print(f"Searching 3x3x3 rank {r3} (gives 3x3x6 rank {r6})...", file=sys.stderr)
        
        result = search_3x3x3(r3, max_iter=30, n_attempts=10, seed=r3)
        if result is not None:
            U3, V3, W3, r3_found = result
            U6, V6, W6, r6_found = kron_decompose(np.array(U3), np.array(V3), np.array(W3), r3_found)
            if _verify(U6, V6, W6, n, m, p):
                if r6_found < best_rank:
                    best_rank = r6_found
                    best_result = {"U": U6, "V": V6, "W": W6}
                    print(f"Found 3x3x6 rank {r6_found} from 3x3x3 rank {r3_found}!", file=sys.stderr)
                    break
        else:
            print(f"3x3x3 rank {r3}: no solution found", file=sys.stderr)
    
    # If we didn't find anything better than schoolbook Kronecker, return best
    if best_result is None:
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
>>>>>>> REPLACE
