    # This is fine in Python (closure).
    
    # Why would it score 1.0?
    # Maybe the `verify` function is slow? No, 84 checks.
    # Maybe the `verify` function returns False?
    # If it returns False, `build_block_strassen` returns None.
    # Then fallback to schoolbook (84).
    # Score = 84/84 = 1.0.
    
    # So the hypothesis is: **The provided code's Strassen implementation is buggy and fails verification.**
    
    # My Strategy: **Fix the Strassen Implementation and Optimize the Tiling.**
    # 1. Implement a robust, verified Strassen block.
    # 2. Use a more optimal tiling strategy (e.g., try different block alignments).
    # 3. Add a deadline guard.
    
    def verify(U, V, W):
        R = U.shape[1]
        # Use numpy broadcasting for speed
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # We need to check T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
        # Loop over i,j,k is 84 iterations.
        # Each iteration involves dot product of 3 vectors of length R.
        # R is ~78. 84 * 78 * 3 ops. Very fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i * m + j, :]
                    vj = V[j * p + k, :]
                    wk = W[k * n + i, :]
                    val = np.dot(ui, vj * wk)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def build_schoolbook():
        U = np.zeros((nm, n*m*p))
        V = np.zeros((mp, n*m*p))
        W = np.zeros((pn, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def build_strassen_block(i0, j0, k0, U_terms, V_terms, W_terms, covered):
        """
        Adds 7 Strassen terms for a 2x2x2 block.
        Verified indices:
        A: i in {i0, i0+1}, j in {j0, j0+1}
        B: j in {j0, j0+1}, k in {k0, k0+1}
        C: k in {k0, k0+1}, i in {i0, i0+1}
        
        Terms:
        M1 = (A00+A11)(B00+B11) -> C00, C11
        M2 = (A10+A11)B00       -> C10
        M3 = A00(B00-B10)       -> C01
        M4 = A11(B11-B01)       -> C00, C10
        M5 = (A00+A01)B11       -> C00(-1), C01(1)
        M6 = (A10-A00)(B00+B01) -> C11
        M7 = (A01-A11)(B10+B11) -> C00
        """
        def add_term(u_dict, v_dict, w_dict):
            u = np.zeros(nm)
            v = np.zeros(mp)
            w = np.zeros(pn)
            for idx, val in u_dict.items():
                u[idx] += val
            for idx, val in v_dict.items():
                v[idx] += val
            for idx, val in w_dict.items():
                w[idx] += val
            U_terms.append(u)
            V_terms.append(v)
            W_terms.append(w)

        # Indices
        i00 = i0*m + j0
        i01 = i0*m + (j0+1)
        i10 = (i0+1)*m + j0
        i11 = (i0+1)*m + (j0+1)
        
        j00 = j0*p + k0
        j01 = j0*p + (k0+1)
        j10 = (j0+1)*p + k0
        j11 = (j0+1)*p + (k0+1)
        
        k00 = k0*n + i0
        k01 = k0*n + (i0+1)
        k10 = (k0+1)*n + i0
        k11 = (k0+1)*n + (i0+1)

        # M1
        add_term({i00:1, i11:1}, {j00:1, j11:1}, {k00:1, k11:1})
        # M2
        add_term({i10:1, i11:1}, {j00:1}, {k10:1})
        # M3
        add_term({i00:1}, {j00:1, j10:-1}, {k01:1})
        # M4
        add_term({i11:1}, {j11:1, j01:-1}, {k00:1, k10:1})
        # M5
        add_term({i00:1, i01:1}, {j11:1}, {k00:-1, k01:1})
        # M6
        add_term({i10:1, i00:-1}, {j00:1, j01:1}, {k11:1})
        # M7
        add_term({i01:1, i11:-1}, {j10:1, j11:1}, {k00:1})
        
        # Mark covered
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))

    def build_block_strassen_optimized():
        U_terms = []
        V_terms = []
        W_terms = []
        covered = set()
        
        # Try different tiling patterns
        # Pattern 1: Standard 2x2x2 tiling
        # i=0,1 can form 2x2 blocks. i=2 is single.
        # j=0,1,2,3. Pairs (0,1), (2,3).
        # k=0..6. Pairs (0,1), (2,3), (4,5). k=6 single.
        
        # Blocks:
        # (0,0,0), (0,2,0), (0,0,2), (0,2,2), (0,0,4), (0,2,4)
        # These cover i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}.
        
        blocks = [
            (0,0,0),
            (0,2,0),
            (0,0,2),
            (0,2,2),
            (0,0,4),
            (0,2,4)
        ]
        
        for b in blocks:
            if time.time() > deadline:
                return None
            build_strassen_block(b[0], b[1], b[2], U_terms, V_terms, W_terms, covered)
            
        # Fill rest with schoolbook
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i,j,k) not in covered:
                        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
                        u[i*m+j] = 1
                        v[j*p+k] = 1
                        w[k*n+i] = 1
                        U_terms.append(u); V_terms.append(v); W_terms.append(w)
                        
        U = np.column_stack(U_terms)
        V = np.column_stack(V_terms)
        W = np.column_stack(W_terms)
        
        if verify(U, V, W):
            return U, V, W
        return None

    # Try optimized block strassen
    result = build_block_strassen_optimized()
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback
    U, V, W = build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
