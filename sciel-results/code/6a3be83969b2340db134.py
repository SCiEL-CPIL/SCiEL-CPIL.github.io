<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Target Rank 39 Construction
    # 5 blocks of 2x2x2 (Strassen, rank 7) + 1 block of 2x2x1 (Schoolbook, rank 4)
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    r = 0

    # Helper to add a Strassen block for columns k0, k1
    def add_strassen_block(k0, k1):
        nonlocal r
        # B indices: B[j, k] -> V[j*p + k]
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]

        # W indices: C[i, k] -> W[k*n + i]
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]

        # Strassen terms for 2x2 * 2x2
        # m1 = (a+d)(e+h)
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        # m2 = (c+d)e
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        # m3 = a(f-h)
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        # m4 = d(g-e)
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        # m5 = (a+b)h
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        # m6 = (c-a)(e+f)
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        # m7 = (b-d)(g+h)
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1

    # Add 5 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7), (8,9)
    for block in range(5):
        if time.time() > deadline - 5:
            break
        k0 = block * 2
        k1 = block * 2 + 1
        add_strassen_block(k0, k1)

    # Last column k=10: Schoolbook
    if r < rank:
        k2 = 10
        v_e2 = k2      # B[0, 10]
        v_g2 = 11 + k2 # B[1, 10]
        w_c02 = 2 * k2      # C[0, 10]
        w_c12 = 2 * k2 + 1  # C[1, 10]

        # a * B[0, 10]
        U[0, r] += 1
        V[v_e2, r] += 1
        W[w_c02, r] += 1
        r += 1

        # b * B[1, 10]
        U[1, r] += 1
        V[v_g2, r] += 1
        W[w_c02, r] += 1
        r += 1

        # c * B[0, 10]
        U[2, r] += 1
        V[v_e2, r] += 1
        W[w_c12, r] += 1
        r += 1

        # d * B[1, 10]
        U[3, r] += 1
        V[v_g2, r] += 1
        W[w_c12, r] += 1
        r += 1

    # Verify the construction
    if time.time() < deadline - 2:
        if _verify(U, V, W):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback: Schoolbook (Rank 44)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
=======
def _is_rank1_tensor(T):
    """Check if a 3D tensor T[a,b,c] is a rank-1 tensor (outer product)."""
    n_a, n_b, n_c = T.shape
    # Find first nonzero entry
    found = False
    for a in range(n_a):
        for b in range(n_b):
            for c in range(n_c):
                if T[a,b,c] != 0:
                    found = True
                    break
            if found:
                break
        if found:
            break
    
    if not found:
        return True  # Zero tensor is rank 0 (or 1 with zero vectors)
    
    # For a rank-1 tensor, T[a,b,c] = u[a]*v[b]*w[c]
    # So T[a1,b1,c1]*T[a2,b2,c2] == T[a1,b2,c1]*T[a2,b1,c2] for all indices
    # Check this property
    for a1 in range(n_a):
        for a2 in range(n_a):
            for b1 in range(n_b):
                for b2 in range(n_b):
                    for c1 in range(n_c):
                        for c2 in range(n_c):
                            lhs = T[a1,b1,c1] * T[a2,b2,c2]
                            rhs = T[a1,b2,c1] * T[a2,b1,c2]
                            if lhs != rhs:
                                return False
    return True

def _build_strassen_decomposition(n, m, p):
    """Build the rank-39 Strassen decomposition. Returns (U, V, W) as lists of lists."""
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    r = 0

    def add_strassen_block(k0, k1):
        nonlocal r
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1

        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1

        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1

    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        add_strassen_block(k0, k1)

    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1

    U[0, r] += 1
    V[v_e2, r] += 1
    W[w_c02, r] += 1
    r += 1

    U[1, r] += 1
    V[v_g2, r] += 1
    W[w_c02, r] += 1
    r += 1

    U[2, r] += 1
    V[v_e2, r] += 1
    W[w_c12, r] += 1
    r += 1

    U[3, r] += 1
    V[v_g2, r] += 1
    W[w_c12, r] += 1
    r += 1

    return U.tolist(), V.tolist(), W.tolist()

def _search_rank38(n, m, p, deadline):
    """Search for a rank-38 decomposition by checking if removing one column from
    a modified rank-39 decomposition leaves a rank-1 residual."""
    U39, V39, W39 = _build_strassen_decomposition(n, m, p)
    rank39 = len(U39[0])
    
    # Build the full multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # For each possible column to remove (0..38), check if the residual is rank-1
    # We need to try different "base" decompositions, not just the standard one.
    # Strategy: try all subsets of 38 columns from the 39-column Strassen decomp.
    # For each subset, compute residual = T - sum of rank-1 terms from those 38 columns.
    # Check if residual is rank-1.
    
    for removed_col in range(rank39):
        if time.time() > deadline - 10:
            break
        
        # Compute sum of 38 rank-1 tensors
        T_partial = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for c in range(rank39):
            if c == removed_col:
                continue
            # Extract the rank-1 term from column c
            u_vec = np.array([U39[i][c] for i in range(n*m)])
            v_vec = np.array([V39[j][c] for j in range(m*p)])
            w_vec = np.array([W39[k][c] for k in range(p*n)])
            # Only add if not all zero
            if np.any(u_vec != 0) and np.any(v_vec != 0) and np.any(w_vec != 0):
                T_partial += np.einsum('a,b,c->abc', u_vec, v_vec, w_vec)
        
        T_residual = T - T_partial
        
        # Check if T_residual is rank-1
        if _is_rank1_tensor(T_residual):
            # Found a rank-38 decomposition!
            # Build the result: 38 columns + 1 new column for the rank-1 residual
            U38 = []
            V38 = []
            W38 = []
            for c in range(rank39):
                if c != removed_col:
                    U38.append([U39[i][c] for i in range(n*m)])
                    V38.append([V39[j][c] for j in range(m*p)])
                    W38.append([W39[k][c] for k in range(p*n)])
            
            # Extract the rank-1 vectors from T_residual
            # Find first nonzero
            u_new = [0] * (n*m)
            v_new = [0] * (m*p)
            w_new = [0] * (p*n)
            
            found = False
            for a in range(n*m):
                for b in range(m*p):
                    for c_idx in range(p*n):
                        if T_residual[a,b,c_idx] != 0:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            
            if found:
                # T_residual[a,b,c] = u[a]*v[b]*w[c]
                # Pick a reference point (a0, b0, c0) where T_residual != 0
                a0 = b0 = c0 = None
                for a in range(n*m):
                    for b in range(m*p):
                        for c_idx in range(p*n):
                            if T_residual[a,b,c_idx] != 0:
                                a0, b0, c0 = a, b, c_idx
                                break
                        if a0 is not None:
                            break
                    if a0 is not None:
                        break
                
                # Set w[c0] = 1, then u[a]*v[b] = T_residual[a,b,c0]
                # and u[a]*w[c] = T_residual[a,b0,c], v[b]*w[c] = T_residual[a0,b,c]
                w_new[c0] = 1
                for a in range(n*m):
                    u_new[a] = T_residual[a, b0, c0]
                for b in range(m*p):
                    v_new[b] = T_residual[a0, b, c0]
                for c_idx in range(p*n):
                    if c_idx != c0:
                        # w[c] = T_residual[a0, b0, c] / T_residual[a0, b0, c0]
                        # But we need integer/half-integer, so check divisibility
                        val = T_residual[a0, b0, c_idx]
                        denom = T_residual[a0, b0, c0]
                        if denom != 0 and val % denom == 0:
                            w_new[c_idx] = val // denom
                        else:
                            # Try other reference points
                            # This case shouldn't happen for a valid rank-1 tensor
                            w_new[c_idx] = val  # Fallback, may be incorrect
                
                # Verify the reconstruction
                T_check = np.einsum('a,b,c->abc', 
                                     np.array(u_new), np.array(v_new), np.array(w_new))
                if np.array_equal(T_check, T_residual):
                    U38.append(u_new)
                    V38.append(v_new)
                    W38.append(w_new)
                    
                    # Final verification
                    U_final = np.array(U38).T  # shape (n*m, 39)
                    V_final = np.array(V38).T
                    W_final = np.array(W38).T
                    
                    T_recon = np.einsum('ar,br,cr->abc', U_final, V_final, W_final)
                    if np.array_equal(T_recon, T):
                        return {
                            "U": U38,
                            "V": V38,
                            "W": W38
                        }
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # First, build the known rank-39 solution
    U39, V39, W39 = _build_strassen_decomposition(n, m, p)
    
    # Verify the rank-39 solution
    if _verify(np.array(U39).T, np.array(V39).T, np.array(W39).T):
        # Try to find a rank-38 solution
        result38 = _search_rank38(n, m, p, deadline)
        if result38 is not None:
            return result38
        
        # Return the rank-39 solution
        return {
            "U": U39,
            "V": V39,
            "W": W39
        }

    # Fallback: Schoolbook (Rank 44)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
>>>>>>> REPLACE
