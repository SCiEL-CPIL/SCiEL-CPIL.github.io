"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank: 19 (Laderman for 4x4x4) + 16 (Schoolbook for 4x4x1) = 35
    rank = 35
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a rank-1 term
    # u_indices: list of (i, j) with coefficients
    # v_indices: list of (j, k) with coefficients
    # w_indices: list of (k, i) with coefficients
    def add_term(r, u_terms, v_terms, w_terms):
        for (i, j), coeff in u_terms:
            U[i * m + j][r] += coeff
        for (j, k), coeff in v_terms:
            V[j * p + k][r] += coeff
        for (k, i), coeff in w_terms:
            W[k * n + i][r] += coeff
            
    # --- Laderman's 19 terms for columns 0-3 ---
    # Indices are 0-based in code, 1-based in formulas
    # a_ij -> A[i, j], b_jk -> B[j, k]
    
    # z1
    u1 = [((0,0),1), ((0,1),1), ((1,0),1), ((1,1),1), ((2,0),1), ((2,1),1), ((3,0),1), ((3,1),1)]
    v1 = [((0,0),1), ((1,0),1), ((2,0),1), ((3,0),1)]
    w1 = [((0,0),1), ((0,1),1), ((0,2),1), ((0,3),1)] # C[0,0] += z1? No, W[k*n+i]
    # Correction: W[k*n+i] corresponds to C[i,k].
    # In Laderman, C[i,k] is a sum of z's.
    # We need the W coefficients for each C[i,k].
    # The formula for C[i,k] is not just 1s.
    
    # Actually, the decomposition T = sum u v w means:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
    
    # So for each Laderman term z_r, we need to know which C[i,k] it contributes to and with what coefficient.
    # Laderman's algorithm defines z_r as products of linear forms.
    # The final C[i,k] is a linear combination of z_r.
    
    # This requires the full Laderman coefficient table for W as well.
    # Since I don't have the W table memorized perfectly, I will use a safer Rank 28 (Strassen) or Rank 19 with a known full implementation.
    
    # Given the risk, I will use the Rank 19 coefficients from a standard source if I can verify.
    # If not, I will fall back to Rank 28 (Strassen on 2x2 blocks).
    
    # Let's implement Rank 28 (Strassen on 2x2 blocks for 8 block products).
    # This is robust and verifiable.
    
    # Reset for Rank 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 coefficients
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # Block indices:
    # A11: i in [0,1], j in [0,1]
    # A12: i in [0,1], j in [2,3]
    # A21: i in [2,3], j in [0,1]
    # A22: i in [2,3], j in [2,3]
    
    # B11: j in [0,1], k in [0,1]
    # B12: j in [0,1], k in [2,3]
    # B21: j in [2,3], k in [0,1]
    # B22: j in [2,3], k in [2,3]
    
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # We need to decompose each 2x2 product using Strassen.
    
    def add_strassen_term(r, A_block_i, A_block_j, B_block_j, B_block_k, strassen_idx):
        # A_block_i: 0 or 1 (row block of A)
        # A_block_j: 0 or 1 (col block of A)
        # B_block_j: 0 or 1 (row block of B)
        # B_block_k: 0 or 1 (col block of B)
        
        # Map block indices to global indices
        # A[i, j] where i in [2*bi, 2*bi+1], j in [2*bj, 2*bj+1]
        # B[j, k] where j in [2*bj, 2*bj+1], k in [2*bk, 2*bk+1]
        
        a11_i, a11_j = 2*A_block_i, 2*A_block_j
        a12_i, a12_j = 2*A_block_i, 2*A_block_j+1
        a21_i, a21_j = 2*A_block_i+1, 2*A_block_j
        a22_i, a22_j = 2*A_block_i+1, 2*A_block_j+1
        
        b11_j, b11_k = 2*B_block_j, 2*B_block_k
        b12_j, b12_k = 2*B_block_j, 2*B_block_k+1
        b21_j, b21_k = 2*B_block_j+1, 2*B_block_k
        b22_j, b22_k = 2*B_block_j+1, 2*B_block_k+1
        
        u_terms = []
        v_terms = []
        w_terms = []
        
        if strassen_idx == 0: # M1
            # (a11+a22)(b11+b22)
            u_terms = [((a11_i, a11_j), 1), ((a22_i, a22_j), 1)]
            v_terms = [((b11_j, b11_k), 1), ((b22_j, b22_k), 1)]
        elif strassen_idx == 1: # M2
            # (a21+a22)b11
            u_terms = [((a21_i, a21_j), 1), ((a22_i, a22_j), 1)]
            v_terms = [((b11_j, b11_k), 1)]
        elif strassen_idx == 2: # M3
            # a11(b12-b22)
            u_terms = [((a11_i, a11_j), 1)]
            v_terms = [((b12_j, b12_k), 1), ((b22_j, b22_k), -1)]
        elif strassen_idx == 3: # M4
            # a22(b21-b11)
            u_terms = [((a22_i, a22_j), 1)]
            v_terms = [((b21_j, b21_k), 1), ((b11_j, b11_k), -1)]
        elif strassen_idx == 4: # M5
            # (a11+a12)b22
            u_terms = [((a11_i, a11_j), 1), ((a12_i, a12_j), 1)]
            v_terms = [((b22_j, b22_k), 1)]
        elif strassen_idx == 5: # M6
            # (a21-a11)(b11+b12)
            u_terms = [((a21_i, a21_j), 1), ((a11_i, a11_j), -1)]
            v_terms = [((b11_j, b11_k), 1), ((b12_j, b12_k), 1)]
        elif strassen_idx == 6: # M7
            # (a12-a22)(b21+b22)
            u_terms = [((a12_i, a12_j), 1), ((a22_i, a22_j), -1)]
            v_terms = [((b21_j, b21_k), 1), ((b22_j, b22_k), 1)]
            
        # W terms:
        # The result of this Strassen block contributes to C[2*ci + di, 2*ck + dk]
        # We need to map the Strassen output M1..M7 to the 4 entries of the 2x2 result block.
        
        # The caller will handle W? No, W depends on which C[i,k] this term affects.
        # This function only adds U and V. W must be added by the caller who knows the C block.
        
        return u_terms, v_terms

    # We need a function that adds a full Strassen block for a specific C block.
    def add_strassen_block(r_start, C_block_i, C_block_j, A_block_i, A_block_j, B_block_j, B_block_k):
        # C_block_i, C_block_j: which 2x2 block of C
        # A_block_i, A_block_j: which 2x2 block of A
        # B_block_j, B_block_k: which 2x2 block of B
        
        # Strassen outputs M1..M7
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Global C indices:
        # C[2*Ci + 0, 2*Cj + 0] corresponds to local C11
        # C[2*Ci + 0, 2*Cj + 1] corresponds to local C12
        # C[2*Ci + 1, 2*Cj + 0] corresponds to local C21
        # C[2*Ci + 1, 2*Cj + 1] corresponds to local C22
        
        # We add 7 terms
        for s in range(7):
            r = r_start + s
            u_terms, v_terms = add_strassen_term(r, A_block_i, A_block_j, B_block_j, B_block_k, s)
            
            for (i, j), coeff in u_terms:
                U[i * m + j][r] = coeff
            for (j, k), coeff in v_terms:
                V[j * p + k][r] = coeff
            
            # Determine W coefficients
            # We need to know which local C entry this M_s contributes to.
            w_coeffs = [0, 0, 0, 0] # C11, C12, C21, C22
            
            if s == 0: # M1
                w_coeffs = [1, 0, 0, 1]
            elif s == 1: # M2
                w_coeffs = [0, 0, 1, -1]
            elif s == 2: # M3
                w_coeffs = [0, 1, 0, 1]
            elif s == 3: # M4
                w_coeffs = [1, 0, 1, 0]
            elif s == 4: # M5
                w_coeffs = [-1, 1, 0, 0]
            elif s == 5: # M6
                w_coeffs = [0, 0, 0, 1]
            elif s == 6: # M7
                w_coeffs = [1, 0, 0, 0]
            
            # Map to global W
            # C11 -> C[2*Ci, 2*Cj]
            # C12 -> C[2*Ci, 2*Cj+1]
            # C21 -> C[2*Ci+1, 2*Cj]
            # C22 -> C[2*Ci+1, 2*Cj+1]
            
            for local_idx, coeff in enumerate(w_coeffs):
                if coeff != 0:
                    if local_idx == 0:
                        gi, gk = 2*C_block_i, 2*C_block_j
                    elif local_idx == 1:
                        gi, gk = 2*C_block_i, 2*C_block_j + 1
                    elif local_idx == 2:
                        gi, gk = 2*C_block_i + 1, 2*C_block_j
                    elif local_idx == 3:
                        gi, gk = 2*C_block_i + 1, 2*C_block_j + 1
                    
                    W[gk * n + gi][r] += coeff

    # 8 block products
    # 1. A11 B11 -> C11
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    # 2. A12 B21 -> C11
    add_strassen_block(7, 0, 0, 0, 1, 1, 0)
    # 3. A11 B12 -> C12
    add_strassen_block(14, 0, 1, 0, 0, 0, 1)
    # 4. A12 B22 -> C12
    add_strassen_block(21, 0, 1, 0, 1, 1, 1)
    # 5. A21 B11 -> C21
    # Wait, we only have rank 28. 8 blocks * 7 = 56.
    # I need to increase rank to 56.
    
    # Reset rank
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Re-run with correct rank
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)   # A11 B11 -> C11
    add_strassen_block(7, 0, 0, 0, 1, 1, 0)   # A12 B21 -> C11
    add_strassen_block(14, 0, 1, 0, 0, 0, 1)  # A11 B12 -> C12
    add_strassen_block(21, 0, 1, 0, 1, 1, 1)  # A12 B22 -> C12
    add_strassen_block(28, 1, 0, 1, 0, 0, 0)  # A21 B11 -> C21
    add_strassen_block(35, 1, 0, 1, 1, 1, 0)  # A22 B21 -> C21
    add_strassen_block(42, 1, 1, 1, 0, 0, 1)  # A21 B12 -> C22
    add_strassen_block(49, 1, 1, 1, 1, 1, 1)  # A22 B22 -> C22
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
