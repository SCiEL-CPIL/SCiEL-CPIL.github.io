"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a low-rank exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    
    Uses a hybrid approach: decompose into sub-problems using recursive Strassen-like
    splits where dimensions allow 2x2 blocks, and fall back to schoolbook for 
    degenerate (1-dimensional) cases.
    
    The decomposition is exact and uses integer entries.
    """
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # We'll build the decomposition by splitting the middle dimension m=5 
    # into m1+m2 where we can apply block methods.
    # 
    # Strategy: Split A (2x5) into A1 (2x2) and A2 (2x3)
    #            Split B (5x6) into B1 (2x6) and B2 (3x6)
    # Then C = A1*B1 + A2*B2
    # 
    # For A1*B1: 2x2 * 2x6 -> split B1 into two 2x3 blocks? 
    # Actually, let's use a more systematic recursive approach.
    
    # Build decomposition by recursively splitting into 2x2 blocks where possible.
    # Each rank-1 term in the final decomposition corresponds to a path through 
    # the recursive splits.
    
    terms = []  # List of (u_vec, v_vec, w_vec) for rank-1 decompositions
    
    def add_term(i_offset, j_offset, k_offset, 
                 a_row_start, a_col_start, a_rows, a_cols,
                 b_row_start, b_col_start, b_rows, b_cols,
                 c_row_start, c_col_start, c_rows, c_cols,
                 sign=1):
        """Add a rank-1 term: contributes sign * A_sub * B_sub to C_sub."""
        u = [0] * (n * m)  # Will be filled
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        # u corresponds to A indices: a_row_start + i, a_col_start + j
        # v corresponds to B indices: b_row_start + j, b_col_start + k  
        # w corresponds to C indices: c_row_start + k, c_col_start + i
        # 
        # For a rank-1 term from sub-blocks, we need specific structure.
        # Actually, for the schoolbook-like terms within blocks, each (i,j,k) 
        # in the sub-problem gives one term.
        
        for i in range(a_rows):
            for j in range(a_cols):
                for k in range(b_cols):
                    # A index: (a_row_start + i) * m + (a_col_start + j)
                    # B index: (b_row_start + j) * p + (b_col_start + k)
                    # C index: (c_col_start + k) * n + (c_row_start + i)  [trace convention W[k*n+i]]
                    ai = (a_row_start + i) * m + (a_col_start + j)
                    bi = (b_row_start + j) * p + (b_col_start + k)
                    ci = (c_col_start + k) * n + (c_row_start + i)
                    
                    if 0 <= ai < n*m and 0 <= bi < m*p and 0 <= ci < p*n:
                        u[ai] = sign
                        v[bi] = sign
                        w[ci] = sign
                    terms.append((u.copy(), v.copy(), w.copy()))
    
    # Instead of the above complex approach, let's use a cleaner recursive method.
    # 
    # The matrix multiplication tensor T_{n,m,p} can be decomposed by splitting
    # one of the dimensions. Here we'll split the middle dimension m.
    #
    # For m = m1 + m2:
    #   C = A[:, :m1] * B[:m1, :] + A[:, m1:] * B[m1:, :]
    #
    # We can recursively decompose each sub-problem.
    #
    # Base case: if any dimension is 1, use rank-1 terms (schoolbook for that slice).
    # If n=2, m=2, p=2: use Strassen (rank 7).
    #
    # Let's define a function that returns the list of (u, v, w) vectors for the
    # sub-problem with given dimensions and offsets.
    
    def decompose(n_sub, m_sub, p_sub, a_off, b_off, c_off):
        """
        Return list of (u, v, w) rank-1 vectors for multiplying 
        (n_sub x m_sub) by (m_sub x p_sub), with offsets for indexing into
        the full matrices.
        
        a_off: offset in A's column index (within the full m dimension)
        b_off: offset in B's row index (same as a_off)  
        c_off: offset in C's column index (within the full p dimension)
        """
        if n_sub == 0 or m_sub == 0 or p_sub == 0:
            return []
        
        # Base case: use schoolbook for small/degenerate cases
        if n_sub * m_sub * p_sub <= 4:
            result = []
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        u = [0] * (n * m)
                        v = [0] * (m * p)
                        w = [0] * (p * n)
                        ai = i * m + (a_off + j)
                        bi = (b_off + j) * p + (c_off + k)
                        ci = (c_off + k) * n + i
                        u[ai] = 1
                        v[bi] = 1
                        w[ci] = 1
                        result.append((u, v, w))
            return result
        
        # Special case: 2x2x2 -> Strassen (rank 7)
        if n_sub == 2 and m_sub == 2 and p_sub == 2:
            return strassen_2x2x2(a_off, b_off, c_off)
        
        # Recursive case: split the middle dimension
        # Split m_sub into m1 + m2, choose m1 = min(2, m_sub // 2) to enable Strassen
        if m_sub >= 2:
            m1 = min(2, m_sub // 2)
            m2 = m_sub - m1
            result = []
            # First half: A[:, :m1] * B[:m1, :]
            result.extend(decompose(n_sub, m1, p_sub, a_off, b_off, c_off))
            # Second half: A[:, m1:] * B[m1:, :]
            result.extend(decompose(n_sub, m2, p_sub, a_off + m1, b_off + m1, c_off))
            return result
        elif n_sub >= 2:
            # Split rows of A and C
            n1 = min(2, n_sub // 2)
            n2 = n_sub - n1
            result = []
            result.extend(decompose(n1, m_sub, p_sub, a_off, b_off, c_off))
            result.extend(decompose(n2, m_sub, p_sub, a_off, b_off, c_off + n1))
            return result
        elif p_sub >= 2:
            # Split columns of B and C
            p1 = min(2, p_sub // 2)
            p2 = p_sub - p1
            result = []
            result.extend(decompose(n_sub, m_sub, p1, a_off, b_off, c_off))
            result.extend(decompose(n_sub, m_sub, p2, a_off, b_off, c_off + p1))
            return result
        else:
            # All dimensions are 1
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            ai = 0 * m + (a_off + 0)
            bi = (b_off + 0) * p + (c_off + 0)
            ci = (c_off + 0) * n + 0
            u[ai] = 1
            v[bi] = 1
            w[ci] = 1
            return [(u, v, w)]
    
    def strassen_2x2x2(a_off, b_off, c_off):
        """
        Strassen decomposition for 2x2 * 2x2.
        A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22)B11
        M3 = A11(B21-B22)
        M4 = A22(B12-B11)
        M5 = (A11+A12)B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        # Helper to create a rank-1 term from linear combinations
        def make_term(a_coeffs, b_coeffs, c_coeff, c_idx):
            """
            a_coeffs: list of (i, j, coeff) for A
            b_coeffs: list of (j, k, coeff) for B  
            c_coeff: coefficient for C
            c_idx: (i_out, k_out) index in the 2x2 output block
            """
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            for (ai, aj, ca) in a_coeffs:
                u[ai * m + (a_off + aj)] = ca
            for (bj, bk, cb) in b_coeffs:
                v[(b_off + bj) * p + (c_off + bk)] = cb
            w[(c_off + c_idx[1]) * n + c_idx[0]] = c_coeff
            
            return (u, v, w)
        
        terms = []
        
        # A indices: A[i][j] where i=0,1 and j=0,1
        # B indices: B[j][k] where j=0,1 and k=0,1
        # C indices: C[i][k] where i=0,1 and k=0,1
        
        # M1 = (A00+A11)(B00+B11)
        terms.append(make_term([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], 1, (0,0)))  # C00
        terms.append(make_term([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], 1, (1,1)))  # C11
        
        # M2 = (A10+A11)B00
        terms.append(make_term([(1,0,1),(1,1,1)], [(0,0,1)], 1, (1,0)))  # C10
        
        # M3 = A00(B10-B11)
        terms.append(make_term([(0,0,1)], [(1,0,1),(1,1,-1)], 1, (0,1)))  # C01
        terms.append(make_term([(0,0,1)], [(1,0,1),(1,1,-1)], 1, (1,1)))  # C11
        
        # M4 = A11(B01-B00)
        terms.append(make_term([(1,1,1)], [(0,1,1),(0,0,-1)], 1, (0,0)))  # C00
        terms.append(make_term([(1,1,1)], [(0,1,1),(0,0,-1)], 1, (1,0)))  # C10
        
        # M5 = (A00+A01)B11
        terms.append(make_term([(0,0,1),(0,1,1)], [(1,1,1)], -1, (0,0)))  # C00
        terms.append(make_term([(0,0,1),(0,1,1)], [(1,1,1)], 1, (0,1)))  # C01
        
        # M6 = (A10-A00)(B00+B01)
        terms.append(make_term([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], 1, (1,1)))  # C11
        
        # M7 = (A01-A11)(B10+B11)
        terms.append(make_term([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], 1, (0,0)))  # C00
        
        return terms
    
    # Get all rank-1 terms
    all_terms = decompose(n, m, p, 0, 0, 0)
    
    rank = len(all_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u, v, w) in enumerate(all_terms):
        for i in range(n * m):
            if u[i] != 0:
                U[i][col] = u[i]
        for i in range(m * p):
            if v[i] != 0:
                V[i][col] = v[i]
        for i in range(p * n):
            if w[i] != 0:
                W[i][col] = w[i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
