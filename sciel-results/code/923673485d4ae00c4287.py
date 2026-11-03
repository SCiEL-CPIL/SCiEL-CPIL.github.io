import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 0
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal rank
        # This is inefficient if we append one by one, but rank is small.
        # We will build lists and then pad to final rank?
        # No, we don't know final rank until we finish.
        # We can just append to lists.
        U.append([0] * 1) # Wait, this is wrong. We need column vectors.
        # Better: maintain list of tuples (u, v, w, coeff) and build matrices at end.
        pass
        
    terms = []
    
    def add_strassen_222(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # a_rows, a_cols: list of 2 global indices for A block
        # b_rows, b_cols: list of 2 global indices for B block
        # c_rows, c_cols: list of 2 global indices for C block
        # A indices: (r, c)
        # B indices: (r, c)
        # C indices: (r, c)
        
        # A00 = A[a_rows[0], a_cols[0]]
        # A01 = A[a_rows[0], a_cols[1]]
        # A10 = A[a_rows[1], a_cols[0]]
        # A11 = A[a_rows[1], a_cols[1]]
        
        # B00 = B[b_rows[0], b_cols[0]]
        # B01 = B[b_rows[0], b_cols[1]]
        # B10 = B[b_rows[1], b_cols[0]]
        # B11 = B[b_rows[1], b_cols[1]]
        
        # C00 = C[c_rows[0], c_cols[0]]
        # C01 = C[c_rows[0], c_cols[1]]
        # C10 = C[c_rows[1], c_cols[0]]
        # C11 = C[c_rows[1], c_cols[1]]
        
        a00 = (a_rows[0], a_cols[0])
        a01 = (a_rows[0], a_cols[1])
        a10 = (a_rows[1], a_cols[0])
        a11 = (a_rows[1], a_cols[1])
        
        b00 = (b_rows[0], b_cols[0])
        b01 = (b_rows[0], b_cols[1])
        b10 = (b_rows[1], b_cols[0])
        b11 = (b_rows[1], b_cols[1])
        
        c00 = (c_rows[0], c_cols[0])
        c01 = (c_rows[0], c_cols[1])
        c10 = (c_rows[1], c_cols[0])
        c11 = (c_rows[1], c_cols[1])
        
        # M1 = (A00+A11)(B00+B11)
        terms.append(((a00, 1), (b00, 1), (c00, 1)))
        terms.append(((a00, 1), (b11, 1), (c00, 1)))
        terms.append(((a11, 1), (b00, 1), (c00, 1)))
        terms.append(((a11, 1), (b11, 1), (c00, 1)))
        terms.append(((a00, 1), (b00, 1), (c11, 1)))
        terms.append(((a00, 1), (b11, 1), (c11, 1)))
        terms.append(((a11, 1), (b00, 1), (c11, 1)))
        terms.append(((a11, 1), (b11, 1), (c11, 1)))
        
        # M2 = (A10+A11)B00
        terms.append(((a10, 1), (b00, 1), (c10, 1)))
        terms.append(((a11, 1), (b00, 1), (c10, 1)))
        terms.append(((a10, 1), (b00, 1), (c11, -1)))
        terms.append(((a11, 1), (b00, 1), (c11, -1)))
        
        # M3 = A00(B10-B11)
        terms.append(((a00, 1), (b10, 1), (c01, 1)))
        terms.append(((a00, 1), (b11, -1), (c01, 1)))
        terms.append(((a00, 1), (b10, 1), (c11, 1)))
        terms.append(((a00, 1), (b11, -1), (c11, 1)))
        
        # M4 = A11(B01-B00)
        terms.append(((a11, 1), (b01, 1), (c00, 1)))
        terms.append(((a11, 1), (b00, -1), (c00, 1)))
        terms.append(((a11, 1), (b01, 1), (c10, 1)))
        terms.append(((a11, 1), (b00, -1), (c10, 1)))
        
        # M5 = (A00+A01)B11
        terms.append(((a00, 1), (b11, 1), (c00, -1)))
        terms.append(((a01, 1), (b11, 1), (c00, -1)))
        terms.append(((a00, 1), (b11, 1), (c01, 1)))
        terms.append(((a01, 1), (b11, 1), (c01, 1)))
        
        # M6 = (A10-A00)(B00+B01)
        terms.append(((a10, 1), (b00, 1), (c11, 1)))
        terms.append(((a10, 1), (b01, 1), (c11, 1)))
        terms.append(((a00, -1), (b00, 1), (c11, 1)))
        terms.append(((a00, -1), (b01, 1), (c11, 1)))
        
        # M7 = (A01-A11)(B10+B11)
        terms.append(((a01, 1), (b10, 1), (c00, 1)))
        terms.append(((a01, 1), (b11, 1), (c00, 1)))
        terms.append(((a11, -1), (b10, 1), (c00, 1)))
        terms.append(((a11, -1), (b11, 1), (c00, 1)))
        
    def add_schoolbook(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # A is |a_rows| x |a_cols|
        # B is |b_rows| x |b_cols|
        # C is |c_rows| x |c_cols|
        # Assumes |a_cols| == |b_rows|
        for i in a_rows:
            for k in a_cols:
                for j in b_cols:
                    # A[i,k] * B[k,j] -> C[i,j]
                    # Note: c_rows and c_cols must correspond to i and j
                    # We assume c_rows[i_idx] = i and c_cols[j_idx] = j
                    # But we pass lists. We need mapping.
                    # For now, assume lists are aligned.
                    # i is global row of A
                    # k is global col of A / row of B
                    # j is global col of B
                    # c_row should be i, c_col should be j
                    
                    # Find index of i in c_rows
                    # This is inefficient. Let's pass c_rows, c_cols and assume 1-to-1 mapping if lengths match.
                    # In our cases:
                    # 2x1x1x2: A rows 0,1; Col 4. B Row 4; Cols 0,1. C Rows 0,1; Cols 0,1.
                    # i in 0,1. k=4. j in 0,1.
                    # c_row is i. c_col is j.
                    pass

    # We need a better way to add terms.
    # Term: (u_idx, coeff_u), (v_idx, coeff_v), (w_idx, coeff_w)
    # u_idx is index in U (row of A flattened? No, U is n*m x rank).
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    
    # My terms list stores (a_idx, b_idx, c_idx, coeff)
    # a_idx is (i, j) for A
    # b_idx is (j, k) for B
    # c_idx is (k, i) for C ? No.
    # C is i x k.
    # W index is k*n + i.
    
    # Let's restart the term accumulation with global indices.
    
    terms = []
    
    def add_term_global(i, j, k, coeff):
        # A[i, j] * B[j, k] contributes to C[i, k]
        terms.append(((i, j), (j, k), (k, i), coeff))
        
    # This is schoolbook.
    # For Strassen, I need to add 7 terms per block.
    # Each Strassen term is a linear combination of A and B entries.
    # e.g. M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # So M1 corresponds to 4 schoolbook terms, all with coeff +1, and they all contribute to C00 (coeff 1) and C11 (coeff 1).
    # Wait.
    # The decomposition is:
    # Sum_t U_t V_t W_t
    # Each term t is a product of a linear form in A, a linear form in B, and a linear form in C (transposed?).
    # Actually, the tensor is T = Sum_{i,j,k} T_{ijk} e_i e_j e_k.
    # We want T = Sum_t u_t \otimes v_t \otimes w_t.
    # u_t is a vector in A-space. v_t in B-space. w_t in C-space.
    # If u_t has non-zero at (i1, j1) and (i2, j2) with coeffs c1, c2...
    # Then the term contributes c1 * c_v * w_{i1, k...}
    
    # So, for a Strassen block:
    # u_1 has A00:1, A11:1.
    # v_1 has B00:1, B11:1.
    # w_1 has C00:1, C11:1.
    # This single term "1" in the decomposition covers 4 products.
    
    # So I should define the 7 vectors (u, v, w) for Strassen directly.
    
    def add_strassen_vectors(a_r, a_c, b_r, b_c, c_r, c_c):
        # a_r, a_c: [0,1] offsets or global indices? Global indices.
        # a_r = [r0, r1], a_c = [c0, c1]
        # A00 = (a_r[0], a_c[0])
        # A01 = (a_r[0], a_c[1])
        # A10 = (a_r[1], a_c[0])
        # A11 = (a_r[1], a_c[1])
        
        # B00 = (b_r[0], b_c[0])
        # B01 = (b_r[0], b_c[1])
        # B10 = (b_r[1], b_c[0])
        # B11 = (b_r[1], b_c[1])
        
        # C00 = (c_r[0], c_c[0]) -> W index c_c[0]*n + c_r[0] ?
        # W index is k*n + i. Here i is row of C, k is col of C.
        # So C[i, k] -> W[k*n + i].
        
        a00 = (a_r[0], a_c[0])
        a01 = (a_r[0], a_c[1])
        a10 = (a_r[1], a_c[0])
        a11 = (a_r[1], a_c[1])
        
        b00 = (b_r[0], b_c[0])
        b01 = (b_r[0], b_c[1])
        b10 = (b_r[1], b_c[0])
        b11 = (b_r[1], b_c[1])
        
        c00 = (c_r[0], c_c[0])
        c01 = (c_r[0], c_c[1])
        c10 = (c_r[1], c_c[0])
        c11 = (c_r[1], c_c[1])
        
        # M1
        u = {a00: 1, a11: 1}
        v = {b00: 1, b11: 1}
        w = {c00: 1, c11: 1}
        terms.append((u, v, w))
        
        # M2
        u = {a10: 1, a11: 1}
        v = {b00: 1}
        w = {c10: 1, c11: -1}
        terms.append((u, v, w))
        
        # M3
        u = {a00: 1}
        v = {b10: 1, b11: -1}
        w = {c01: 1, c11: 1}
        terms.append((u, v, w))
        
        # M4
        u = {a11: 1}
        v = {b01: 1, b00: -1}
        w = {c00: 1, c10: 1}
        terms.append((u, v, w))
        
        # M5
        u = {a00: 1, a01: 1}
        v = {b11: 1}
        w = {c00: -1, c01: 1}
        terms.append((u, v, w))
        
        # M6
        u = {a10: 1, a00: -1}
        v = {b00: 1, b01: 1}
        w = {c11: 1}
        terms.append((u, v, w))
        
        # M7
        u = {a01: 1, a11: -1}
        v = {b10: 1, b11: 1}
        w = {c00: 1}
        terms.append((u, v, w))
        
    def add_schoolbook_vectors(a_r, a_c, b_r, b_c, c_r, c_c):
        # For simple blocks, we can just add schoolbook terms.
        # Or use the 1x1x1 logic.
        # If a_c len 1, b_r len 1, it's outer product?
        # No, it's just products.
        for i_idx, i in enumerate(a_r):
            for k_idx, k in enumerate(a_c):
                for j_idx, j in enumerate(b_c):
                    # A[i, k] B[k, j] -> C[i, j]
                    # W index: j*n + i
                    u = {(i, k): 1}
                    v = {(k, j): 1}
                    w = {(i, j): 1} # Note: w is indexed by (row, col) of C?
                    # In my term tuple (u, v, w), w is a dict mapping C-index to coeff.
                    # C index is (row, col).
                    terms.append((u, v, w))
                    
    # Define blocks
    # A: 3x5. Rows 0,1,2. Cols 0,1,2,3,4.
    # B: 5x6. Rows 0,1,2,3,4. Cols 0,1,2,3,4,5.
    # C: 3x6. Rows 0,1,2. Cols 0,1,2,3,4,5.
    
    # Block 1: A[0:2, 0:2] B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_vectors([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # Block 2: A[0:2, 2:4] B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_vectors([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    
    # Block 3: A[0:2, 4:5] B[4:5, 0:2] -> C[0:2, 0:2]
    add_schoolbook_vectors([0,1], [4], [4], [0,1], [0,1], [0,1])
    
    # Block 4: A[0:2, 0:2] B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_vectors([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    
    # Block 5: A[0:2, 2:4] B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_vectors([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    
    # Block 6: A[0:2, 4:5] B[4:5, 2:4] -> C[0:2, 2:4]
    add_schoolbook_vectors([0,1], [4], [4], [2,3], [0,1], [2,3])
    
    # Block 7: A[0:2, 0:2] B[0:2, 4:6] -> C[0:2, 4:6]
    add_strassen_vectors([0,1], [0,1], [0,1], [4,5], [0,1], [4,5])
    
    # Block 8: A[0:2, 2:4] B[2:4, 4:6] -> C[0:2, 4:6]
    add_strassen_vectors([0,1], [2,3], [2,3], [4,5], [0,1], [4,5])
    
    # Block 9: A[0:2, 4:5] B[4:5, 4:6] -> C[0:2, 4:6]
    add_schoolbook_vectors([0,1], [4], [4], [4,5], [0,1], [4,5])
    
    # Block 10: A[2:3, 0:2] B[0:2, 0:2] -> C[2:3, 0:2]
    add_schoolbook_vectors([2], [0,1], [0,1], [0,1], [2], [0,1])
    
    # Block 11: A[2:3, 2:4] B[2:4, 0:2] -> C[2:3, 0:2]
    add_schoolbook_vectors([2], [2,3], [2,3], [0,1], [2], [0,1])
    
    # Block 12: A[2:3, 4:5] B[4:5, 0:2] -> C[2:3, 0:2]
    add_schoolbook_vectors([2], [4], [4], [0,1], [2], [0,1])
    
    # Block 13: A[2:3, 0:2] B[0:2, 2:4] -> C[2:3, 2:4]
    add_schoolbook_vectors([2], [0,1], [0,1], [2,3], [2], [2,3])
    
    # Block 14: A[2:3, 2:4] B[2:4, 2:4] -> C[2:3, 2:4]
    add_schoolbook_vectors([2], [2,3], [2,3], [2,3], [2], [2,3])
    
    # Block 15: A[2:3, 4:5] B[4:5, 2:4] -> C[2:3, 2:4]
    add_schoolbook_vectors([2], [4], [4], [2,3], [2], [2,3])
    
    # Block 16: A[2:3, 0:2] B[0:2, 4:6] -> C[2:3, 4:6]
    add_schoolbook_vectors([2], [0,1], [0,1], [4,5], [2], [4,5])
    
    # Block 17: A[2:3, 2:4] B[2:4, 4:6] -> C[2:3, 4:6]
    add_schoolbook_vectors([2], [2,3], [2,3], [4,5], [2], [4,5])
    
    # Block 18: A[2:3, 4:5] B[4:5, 4:6] -> C[2:3, 4:6]
    add_schoolbook_vectors([2], [4], [4], [4,5], [2], [4,5])
    
    # Build Matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][t_idx] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][t_idx] = coeff
        for (i, k), coeff in w_dict.items():
            # W index is k*n + i
            W[k * n + i][t_idx] = coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
