import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition
    # We generate the 49 factors directly.
    # The decomposition is based on applying Strassen's algorithm to 4x4 matrices
    # by splitting them into 2x2 blocks, and then applying Strassen to each 2x2 block product.
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to get global index for a 2x2 block element
    # Block (br, bc) in 4x4 matrix. Element (lr, lc) in 2x2 block.
    # Global row = br*2 + lr, Global col = bc*2 + lc.
    # Flatten index for U (A side, n*m): row*n + col. Here n=4.
    # Flatten index for V (B side, m*p): row*m + col. Here m=4.
    # Flatten index for W (C side, p*n): col*n + row. Here n=4.
    
    def idx_A(br, bc, lr, lc):
        r = br * 2 + lr
        c = bc * 2 + lc
        return r * 4 + c

    def idx_B(br, bc, lr, lc):
        r = br * 2 + lr
        c = bc * 2 + lc
        return r * 4 + c

    def idx_C(br, bc, lr, lc):
        # C is n x p. Output index in W is k*n + i, where i is row, k is col.
        # Block (br, bc) corresponds to C rows br*2..br*2+1, cols bc*2..bc*2+1.
        r = br * 2 + lr
        c = bc * 2 + lc
        return c * 4 + r

    # Strassen factors for 2x2 matrix multiplication
    # U2, V2, W2 are 4x7 matrices.
    # Indices 0..3 correspond to elements 00, 01, 10, 11 of the 2x2 matrix.
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    # U2: A00=1, A11=1
    # V2: B00=1, B11=1
    # W2: C00+=1, C11+=1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    # U2: A10=1, A11=1
    # V2: B00=1
    # W2: C10+=1, C11-=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3 = A00(B01-B11)
    # U2: A00=1
    # V2: B01=1, B11=-1
    # W2: C01+=1, C11+=1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4 = A11(B10-B00)
    # U2: A11=1
    # V2: B10=1, B00=-1
    # W2: C00+=1, C10+=1
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5 = (A00+A01)B11
    # U2: A00=1, A01=1
    # V2: B11=1
    # W2: C00-=1, C01+=1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # U2: A10=1, A00=-1
    # V2: B00=1, B01=1
    # W2: C11+=1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # U2: A01=1, A11=-1
    # V2: B10=1, B11=1
    # W2: C00+=1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Outer Strassen terms for 4x4
    # There are 7 terms M_outer_k.
    # Each M_outer_k is P_k * Q_k, where P_k and Q_k are 2x2 blocks formed from A and B blocks.
    # P_k = sum (alpha_ij * A_ij)
    # Q_k = sum (beta_ij * B_ij)
    # M_outer_k contributes to C blocks.
    
    # Term 1: (A00 + A11)(B00 + B11)
    # P1 = A00 + A11
    # Q1 = B00 + B11
    # C00 += M1, C11 += M1
    # Term 2: (A10 + A11)B00
    # P2 = A10 + A11
    # Q2 = B00
    # C10 += M2, C11 -= M2
    # Term 3: A00(B01 - B11)
    # P3 = A00
    # Q3 = B01 - B11
    # C01 += M3, C11 += M3
    # Term 4: A11(B10 - B00)
    # P4 = A11
    # Q4 = B10 - B00
    # C00 += M4, C10 += M4
    # Term 5: (A00 + A01)B11
    # P5 = A00 + A01
    # Q5 = B11
    # C00 -= M5, C01 += M5
    # Term 6: (A10 - A00)(B00 + B01)
    # P6 = A10 - A00
    # Q6 = B00 + B01
    # C11 += M6
    # Term 7: (A01 - A11)(B10 + B11)
    # P7 = A01 - A11
    # Q7 = B10 + B11
    # C00 += M7
    
    # Structure:
    # outer_k: 
    #   P_blocks: list of (block_row, block_col, coeff)
    #   Q_blocks: list of (block_row, block_col, coeff)
    #   C_contribs: list of (block_row, block_col, coeff)
    
    outer_terms = [
        # k=0
        {
            'P': [(0, 0, 1), (1, 1, 1)],
            'Q': [(0, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1), (1, 1, 1)]
        },
        # k=1
        {
            'P': [(1, 0, 1), (1, 1, 1)],
            'Q': [(0, 0, 1)],
            'C': [(1, 0, 1), (1, 1, -1)]
        },
        # k=2
        {
            'P': [(0, 0, 1)],
            'Q': [(0, 1, 1), (1, 1, -1)],
            'C': [(0, 1, 1), (1, 1, 1)]
        },
        # k=3
        {
            'P': [(1, 1, 1)],
            'Q': [(1, 0, 1), (0, 0, -1)],
            'C': [(0, 0, 1), (1, 0, 1)]
        },
        # k=4
        {
            'P': [(0, 0, 1), (0, 1, 1)],
            'Q': [(1, 1, 1)],
            'C': [(0, 0, -1), (0, 1, 1)]
        },
        # k=5
        {
            'P': [(1, 0, 1), (0, 0, -1)],
            'Q': [(0, 0, 1), (0, 1, 1)],
            'C': [(1, 1, 1)]
        },
        # k=6
        {
            'P': [(0, 1, 1), (1, 1, -1)],
            'Q': [(1, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1)]
        }
    ]
    
    for k in range(7):
        term = outer_terms[k]
        P_blocks = term['P']
        Q_blocks = term['Q']
        C_contribs = term['C']
        
        for s in range(7):
            col_idx = k * 7 + s
            
            # U side
            for local_a in range(4):
                coeff_inner = U2[local_a, s]
                if coeff_inner != 0:
                    lr = local_a // 2
                    lc = local_a % 2
                    for br, bc, coeff_block in P_blocks:
                        g_idx = idx_A(br, bc, lr, lc)
                        U[g_idx, col_idx] += coeff_inner * coeff_block
            
            # V side
            for local_b in range(4):
                coeff_inner = V2[local_b, s]
                if coeff_inner != 0:
                    lr = local_b // 2
                    lc = local_b % 2
                    for br, bc, coeff_block in Q_blocks:
                        g_idx = idx_B(br, bc, lr, lc)
                        V[g_idx, col_idx] += coeff_inner * coeff_block
            
            # W side
            for local_c in range(4):
                coeff_inner = W2[local_c, s]
                if coeff_inner != 0:
                    lr = local_c // 2
                    lc = local_c % 2
                    for br, bc, coeff_block in C_contribs:
                        g_idx = idx_C(br, bc, lr, lc)
                        W[g_idx, col_idx] += coeff_inner * coeff_block
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
