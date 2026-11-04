"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition recursively
    # Rank estimate:
    # R(3, 3, 7) <= R(2, 3, 7) + R(1, 3, 7)
    # R(1, 3, 7) = 21
    # R(2, 3, 7) <= R(2, 2, 7) + R(2, 1, 7)
    # R(2, 1, 7) = 14
    # R(2, 2, 7) <= R(2, 2, 3) + R(2, 2, 4)
    # R(2, 2, 4) = 2 * R(2, 2, 2) = 14
    # R(2, 2, 3) <= R(2, 2, 2) + R(2, 2, 1) = 7 + 4 = 11
    # Total = 21 + 14 + 11 + 14 = 60
    
    # We will store U, V, W as lists of lists
    # But we don't know the final rank until we build it.
    # So we will accumulate columns.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(n_sub, m_sub, p_sub, i0, j0, k0, decomposition_func):
        """
        Adds a block decomposition for A[i0:i0+n_sub, j0:j0+m_sub] * B[j0:j0+m_sub, k0:k0+p_sub]
        decomposition_func should return a list of tuples (u_vals, v_vals, w_vals)
        where u_vals is a dict { (i, j): val }, etc.
        """
        # Get the decomposition
        terms = decomposition_func()
        for term in terms:
            u_dict, v_dict, w_dict = term
            # Create full vectors for this term
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            for (li, lj), val in u_dict.items():
                gi = i0 + li
                gj = j0 + lj
                idx = gi * m + gj
                u_vec[idx] += val
                
            for (lj, lk), val in v_dict.items():
                gj = j0 + lj
                gk = k0 + lk
                idx = gj * p + gk
                v_vec[idx] += val
                
            for (lk, li), val in w_dict.items():
                gk = k0 + lk
                gi = i0 + li
                idx = gk * n + gi
                w_vec[idx] += val
                
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)

    def schoolbook(n_s, m_s, p_s):
        """Returns list of terms for schoolbook decomposition of n_s x m_s * m_s x p_s"""
        terms = []
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))
        return terms

    def strassen_222():
        """Returns list of terms for Strassen decomposition of 2x2 * 2x2"""
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We define each M as a term (u, v, w)
        # u, v, w are dicts mapping local indices to coefficients
        
        # M1
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1} # Wait, W maps (k, i) to output.
        # Output C11 (0,0) gets M1. Output C22 (1,1) gets M1.
        # So w1 should be {(0,0): 1, (1,1): 1} ?
        # W index is (k, i).
        # C11 is i=0, k=0. C22 is i=1, k=1.
        # So yes.
        
        # M2
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): 1} # C21 (i=1, k=0) and C22 (i=1, k=1)?
        # C21 = M2 + M4. C21 is i=1, k=0.
        # C22 = M1 - M2 + M3 + M6. C22 is i=1, k=1.
        # So M2 contributes to C21 (i=1, k=0) and C22 (i=1, k=1).
        # Wait, M2 = (A21 + A22)B11.
        # A21 is (1,0), A22 is (1,1). B11 is (0,0).
        # Product is scalar.
        # C21 = A21 B11 + A22 B21.
        # M2 = A21 B11 + A22 B11.
        # This doesn't look like C21 directly.
        # Let's re-verify Strassen formulas.
        
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's map M2 to outputs.
        # M2 contributes +1 to C21 and -1 to C22.
        # C21 is (i=1, k=0). C22 is (i=1, k=1).
        # So w2 = {(0,1): 1, (1,1): -1}.
        
        # M3
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        # M3 contributes +1 to C12 and +1 to C22.
        # C12 is (i=0, k=1). C22 is (i=1, k=1).
        w3 = {(1,0): 1, (1,1): 1}
        
        # M4
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        # M4 contributes +1 to C11 and +1 to C21.
        # C11 is (i=0, k=0). C21 is (i=1, k=0).
        w4 = {(0,0): 1, (0,1): 1}
        
        # M5
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        # M5 contributes -1 to C11 and +1 to C12.
        # C11 is (i=0, k=0). C12 is (i=0, k=1).
        w5 = {(0,0): -1, (1,0): 1}
        
        # M6
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        # M6 contributes +1 to C22.
        # C22 is (i=1, k=1).
        w6 = {(1,1): 1}
        
        # M7
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        # M7 contributes +1 to C11.
        # C11 is (i=0, k=0).
        w7 = {(0,0): 1}
        
        return [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]

    def combine_decomps(decomp1, decomp2):
        """Concatenates two lists of terms"""
        return decomp1 + decomp2

    # Build R(2, 2, 3)
    # Split B (2x3) into B1 (2x2) and B2 (2x1)
    # A (2x2) * B1 (2x2) -> Strassen
    # A (2x2) * B2 (2x1) -> Schoolbook
    def r_223():
        # B1 is cols 0,1. B2 is col 2.
        # We need to handle the indices correctly in the caller.
        # Here we just return the terms for a local 2x2x3 multiplication.
        # Local A: 2x2. Local B: 2x3.
        # Term 1: A * B[:, 0:2] (2x2x2)
        # Term 2: A * B[:, 2:3] (2x2x1)
        
        # For Term 1, B is 2x2. Indices j in 0,1, k in 0,1.
        # For Term 2, B is 2x1. Indices j in 0,1, k in 2.
        
        # We will generate terms for 2x2x2 and 2x2x1 and adjust k indices.
        
        terms_222 = strassen_222()
        # Adjust k indices for 2x2x2 part (k 0,1 -> 0,1)
        # No change needed if we treat it as sub-block k=0,1.
        
        terms_221 = schoolbook(2, 2, 1)
        # Adjust k indices for 2x2x1 part (k 0 -> 2)
        adjusted_221 = []
        for u, v, w in terms_221:
            new_w = {}
            for (lk, li), val in w.items():
                new_w[(lk + 2, li)] = val
            adjusted_221.append((u, v, new_w))
            
        return combine_decomps(terms_222, adjusted_221)

    # Build R(2, 2, 4)
    # Split B (2x4) into B1 (2x2) and B2 (2x2)
    def r_224():
        terms_1 = strassen_222()
        # k 0,1 -> 0,1
        terms_2 = strassen_222()
        # k 0,1 -> 2,3
        adjusted_2 = []
        for u, v, w in terms_2:
            new_w = {}
            for (lk, li), val in w.items():
                new_w[(lk + 2, li)] = val
            adjusted_2.append((u, v, new_w))
        return combine_decomps(terms_1, adjusted_2)

    # Build R(2, 2, 7)
    # Split B (2x7) into B1 (2x3) and B2 (2x4)
    def r_227():
        terms_1 = r_223()
        # k 0,1,2 -> 0,1,2
        terms_2 = r_224()
        # k 0,1,2,3 -> 3,4,5,6
        adjusted_2 = []
        for u, v, w in terms_2:
            new_w = {}
            for (lk, li), val in w.items():
                new_w[(lk + 3, li)] = val
            adjusted_2.append((u, v, new_w))
        return combine_decomps(terms_1, adjusted_2)

    # Build R(2, 1, 7)
    # Schoolbook
    def r_217():
        return schoolbook(2, 1, 7)

    # Build R(2, 3, 7)
    # Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x7) into B1 (2x7) and B2 (1x7)
    # A1 * B1 -> R(2, 2, 7)
    # A2 * B2 -> R(2, 1, 7)
    def r_237():
        terms_1 = r_227()
        # A1 is rows 0,1, cols 0,1. B1 is rows 0,1, cols 0,6.
        # Local indices for A1: i 0,1; j 0,1.
        # Local indices for B1: j 0,1; k 0,6.
        # This matches r_227 local indices.
        
        terms_2 = r_217()
        # A2 is rows 0,1, col 2. B2 is row 2, cols 0,6.
        # Local indices for A2: i 0,1; j 0. (Mapped from global j=2)
        # Local indices for B2: j 0; k 0,6. (Mapped from global j=2)
        # r_217 uses local j=0.
        # We need to shift j indices in V for A2/B2 part?
        # In add_block, we pass j0.
        # For A1/B1, i0=0, j0=0, k0=0.
        # For A2/B2, i0=0, j0=2, k0=0.
        # So we just call add_block twice.
        
        # But here we are building the decomposition function.
        # We need to combine terms.
        # The terms from r_227 have j in 0,1.
        # The terms from r_217 have j in 0.
        # We need to shift j for the second part to 1?
        # No, in r_217, local j is 0.
        # In the global context of R(2, 3, 7), A is 2x3.
        # A1 is cols 0,1. A2 is col 2.
        # So for A2, local j=0 corresponds to global j=2.
        # So we shift j by 2?
        # Wait, add_block takes j0.
        # If we use add_block(2, 2, 7, 0, 0, 0, r_227), it handles j0=0.
        # If we use add_block(2, 1, 7, 0, 2, 0, r_217), it handles j0=2.
        # So we don't need to shift indices inside the decomposition function.
        # We just need to call add_block correctly.
        
        # However, this function is supposed to return terms for a local 2x3x7.
        # So we must shift indices to be local to 2x3x7.
        # Local j for A1 is 0,1.
        # Local j for A2 is 2.
        # So we shift j in r_217 terms by 2.
        
        adjusted_2 = []
        for u, v, w in terms_2:
            new_v = {}
            for (lj, lk), val in v.items():
                new_v[(lj + 2, lk)] = val
            # u indices: local j for A is 0. Global j is 2.
            # So we shift j in u by 2.
            new_u = {}
            for (li, lj), val in u.items():
                new_u[(li, lj + 2)] = val
            adjusted_2.append((new_u, new_v, w))
            
        return combine_decomps(terms_1, adjusted_2)

    # Build R(1, 3, 7)
    # Schoolbook
    def r_137():
        return schoolbook(1, 3, 7)

    # Build R(3, 3, 7)
    # Split A (3x3) into A1 (2x3) and A2 (1x3)
    # A1 * B (3x7) -> R(2, 3, 7)
    # A2 * B (3x7) -> R(1, 3, 7)
    
    # Add block for A1 (rows 0,1)
    add_block(2, 3, 7, 0, 0, 0, r_237)
    # Add block for A2 (row 2)
    add_block(1, 3, 7, 2, 0, 0, r_137)
    
    # Convert lists to matrices
    rank = len(U_cols)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n * m):
            U[i][c] = U_cols[c][i]
        for i in range(m * p):
            V[i][c] = V_cols[c][i]
        for i in range(p * n):
            W[i][c] = W_cols[c][i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
