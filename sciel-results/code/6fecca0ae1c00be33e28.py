# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # (U_coeffs, V_coeffs, W_coeffs)
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Define the 4 blocks of the 2x4x4 multiplication
    # Block k corresponds to:
    # A_sub: rows 0-1, cols [2k, 2k+1]
    # B_sub: rows [2k, 2k+1], cols [2k, 2k+1] ? No.
    
    # Let's map the 4x4 B and 2x4 A to 2x2 blocks.
    # A (2x4) = [A_0, A_1] where A_0, A_1 are 2x2.
    # B (4x4) = [[B_00, B_01], [B_10, B_11]] where B_ij are 2x2.
    # C (2x4) = [C_0, C_1] where C_0, C_1 are 2x2.
    # C_0 = A_0 B_00 + A_1 B_10
    # C_1 = A_0 B_01 + A_1 B_11
    
    # We need to decompose C_0 and C_1.
    # C_0 involves products: A_0 B_00 and A_1 B_10.
    # C_1 involves products: A_0 B_01 and A_1 B_11.
    
    # We can group the Strassen terms.
    # For C_0:
    # Term 1: A_0 * B_00 (7 mults)
    # Term 2: A_1 * B_10 (7 mults)
    # For C_1:
    # Term 3: A_0 * B_01 (7 mults)
    # Term 4: A_1 * B_11 (7 mults)
    
    # Total 28 mults.
    
    # Mapping global indices:
    # A: 2x4. Index i (0,1), j (0,3). Flat: i*4+j.
    # B: 4x4. Index j (0,3), r (0,3). Flat: j*4+r.
    # C: 2x4. Index i (0,1), r (0,3). Flat: r*2+i (Trace convention W[k*n+i,r] -> W[r*2+i]? 
    # Task says: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually W is p x n. Here p=4, n=2.
    # W index: k (0..p-1), i (0..n-1).
    # If W is flattened, is it row-major?
    # "W[k*n+i,r]" suggests W is a matrix of shape (p*n, rank)?
    # No, "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)".
    # So W has shape (4*2, 28) = (8, 28).
    # The index in the flattened W is "k*n+i".
    # k is the row of C? No, C is 2x4.
    # C[i, r] where i in 0..1, r in 0..3.
    # The tensor is T[i, j, r] = C[i, r] = sum_k A[i, k] B[k, r].
    # The factor W corresponds to the output indices.
    # Shape (p*n, rank) = (4*2, rank).
    # The index in W is likely (r, i) or (i, r)?
    # "W[k*n+i,r]" in the description "output factor follows ... W[k*n+i,r]" is confusing.
    # It probably means the element at row (k*n+i) and column r of the W matrix.
    # If W has shape (p*n, rank), the rows are indexed by the output tensor indices.
    # Output tensor C is 2x4. Indices (i, r) with i in 0..1, r in 0..3.
    # If we flatten C to a vector of length 8, how?
    # Standard row-major: C[i, r] -> i*4 + r.
    # But the shape is (p*n, rank) = (4*2, rank).
    # If the index is k*n+i, with n=2, k in 0..3 (p), i in 0..1 (n).
    # Then index = k*2 + i.
    # This corresponds to C[i, k]?
    # Let's check the trace convention.
    # Trace(C) = sum C[i, i]? No, dimensions don't match.
    # "official trace convention" usually refers to how the tensor is defined.
    # T[i, j, k] = C[i, k] where C = A B.
    # A is n x m. B is m x p. C is n x p.
    # Here n=2, m=4, p=4.
    # C is 2 x 4.
    # W shape is (p*n, rank) = (8, rank).
    # The row index of W corresponds to the (i, r) pair of C.
    # If the index is k*n+i, and k ranges 0..p-1, i ranges 0..n-1.
    # Then row = r*2 + i.
    # This implies C is stored as C[i, r] -> index r*2 + i.
    # This is column-major for C (if C is viewed as n x p, transposed to p x n?).
    # Let's assume W[r*2 + i, s] contributes to C[i, r].
    
    # Let's verify with the current program's get_global_C.
    # Current: return r * 2 + i.
    # So W index is r*2 + i.
    
    # Let's define the blocks for A, B, C.
    # A_0: rows 0-1, cols 0-1. Global A idx: i*4 + j.
    #   i in 0,1. j in 0,1.
    # A_1: rows 0-1, cols 2-3. Global A idx: i*4 + (2+j).
    
    # B_00: rows 0-1, cols 0-1. Global B idx: j*4 + r.
    #   j in 0,1. r in 0,1.
    # B_10: rows 2-3, cols 0-1. Global B idx: (2+j)*4 + r.
    #   j in 0,1. r in 0,1.
    # B_01: rows 0-1, cols 2-3. Global B idx: j*4 + (2+r).
    #   j in 0,1. r in 0,1.
    # B_11: rows 2-3, cols 2-3. Global B idx: (2+j)*4 + (2+r).
    #   j in 0,1. r in 0,1.
    
    # C_0: rows 0-1, cols 0-1. Global C idx: r*2 + i.
    #   i in 0,1. r in 0,1.
    # C_1: rows 0-1, cols 2-3. Global C idx: (2+r)*2 + i.
    #   i in 0,1. r in 0,1.
    
    def add_term(col_offset, a_block, b_block, c_block, term_idx):
        u_c, v_c, w_c = terms[term_idx]
        
        # Map local 2x2 indices to global
        # Local (li, lj) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        for l_idx, coeff in u_c.items():
            li = l_idx // 2
            lj = l_idx % 2
            if a_block == 0: # A_0
                i, j = li, lj
            elif a_block == 1: # A_1
                i, j = li, 2 + lj
            g_idx = i * 4 + j
            U[g_idx][col_offset] += coeff
            
        for l_idx, coeff in v_c.items():
            li = l_idx // 2
            lj = l_idx % 2
            if b_block == 0: # B_00
                j, r = li, lj
            elif b_block == 1: # B_10
                j, r = 2 + li, lj
            elif b_block == 2: # B_01
                j, r = li, 2 + lj
            elif b_block == 3: # B_11
                j, r = 2 + li, 2 + lj
            g_idx = j * 4 + r
            V[g_idx][col_offset] += coeff
            
        for l_idx, coeff in w_c.items():
            li = l_idx // 2
            lj = l_idx % 2
            # li is i, lj is r
            i, r = li, lj
            if c_block == 0: # C_0
                i_g, r_g = i, r
            elif c_block == 1: # C_1
                i_g, r_g = i, 2 + r
            g_idx = r_g * 2 + i_g
            W[g_idx][col_offset] += coeff

    # C_0 = A_0 B_00 + A_1 B_10
    # C_1 = A_0 B_01 + A_1 B_11
    
    # Block 0: A_0 * B_00 -> C_0
    for s in range(7):
        add_term(s, 0, 0, 0, s)
        
    # Block 1: A_1 * B_10 -> C_0
    for s in range(7):
        add_term(7 + s, 1, 1, 0, s)
        
    # Block 2: A_0 * B_01 -> C_1
    for s in range(7):
        add_term(14 + s, 0, 2, 1, s)
        
    # Block 3: A_1 * B_11 -> C_1
    for s in range(7):
        add_term(21 + s, 1, 3, 1, s)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
