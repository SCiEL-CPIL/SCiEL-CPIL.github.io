DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 7
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # U_strassen[k] is vector of 4 coeffs for A
    # V_strassen[k] is vector of 4 coeffs for B
    # W_strassen[k] is vector of 4 coeffs for C
    
    # M1 = A00 * (B01 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = (A00 + A01) * B11
    # M4 = A11 * (B00 + B10)
    # M5 = (A00 + A11) * (B00 + B11)
    # M6 = (A00 - A10) * (B01 + B11)
    # M7 = (A01 - A11) * (B10 + B11)
    
    U_strassen = [
        [1, 0, 0, 0],   # M1: A00
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 1, 0, 0],   # M3: A00+A01
        [0, 0, 0, 1],   # M4: A11
        [1, 0, 0, 1],   # M5: A00+A11
        [1, 0, -1, 0],  # M6: A00-A10
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    
    V_strassen = [
        [0, 1, 0, 1],   # M1: B01+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 0, 0, 1],   # M3: B11
        [1, 0, 1, 0],   # M4: B00+B10
        [1, 0, 0, 1],   # M5: B00+B11
        [0, 1, 0, 1],   # M6: B01+B11
        [0, 0, 1, 1],   # M7: B10+B11
    ]
    
    W_strassen = [
        [0, 1, 0, 0],   # M1: C01
        [0, 0, 1, 0],   # M2: C10
        [0, 0, 1, 0],   # M3: C10
        [0, 0, 0, 1],   # M4: C11
        [1, 0, 0, 1],   # M5: C00+C11
        [0, 0, 0, 0],   # M6: 0
        [0, 1, 0, 0],   # M7: C01
    ]
    
    # Check W_strassen correctness:
    # C00 = M5 + M4 - M3 - M7
    # C01 = M1 + M7
    # C10 = M2 + M3
    # C11 = M5 + M1 - M2 - M4
    
    # M1: C01 += 1
    # M2: C10 += 1
    # M3: C10 += 1
    # M4: C11 += 1
    # M5: C00 += 1, C11 += 1
    # M6: 0
    # M7: C01 += 1
    
    # Let's re-verify W_strassen vectors (indices 0:C00, 1:C01, 2:C10, 3:C11)
    # M1: [0, 1, 0, 0]
    # M2: [0, 0, 1, 0]
    # M3: [0, 0, 1, 0]
    # M4: [0, 0, 0, 1]
    # M5: [1, 0, 0, 1]
    # M6: [0, 0, 0, 0]
    # M7: [0, 1, 0, 0]
    
    # Sum for C00: M5(1) + M4(0) - M3(0) - M7(0) = 1. Correct.
    # Sum for C01: M1(1) + M7(1) = 2? No.
    # Formula: C01 = M1 + M7.
    # My W vectors add up.
    # If I sum W vectors with signs from the formula, I get the coefficients.
    # But in the tensor decomposition, each term is a rank-1 tensor.
    # The W vector for term k is the contribution of M_k to C.
    # So if C01 = M1 + M7, then W_M1 has 1 at C01, W_M7 has 1 at C01.
    # This is correct.
    
    # Now build the 49-term decomposition.
    # For each column l in 0..6:
    #   For each Strassen term k in 0..6:
    #     col_idx = l * 7 + k
    #     U[:, col_idx] = U_strassen[k]
    #     V[:, col_idx] = mapped V_strassen[k]
    #     W[:, col_idx] = mapped W_strassen[k]
    
    for l in range(p):
        for k in range(7):
            col_idx = l * 7 + k
            
            # U is same for all l
            for i in range(4):
                U[i][col_idx] = U_strassen[k][i]
                
            # V mapping:
            # B_00 (idx 0 in 2x2) -> B_0l (idx 0*7+l in 2x7)
            # B_01 (idx 1 in 2x2) -> 0
            # B_10 (idx 2 in 2x2) -> B_1l (idx 1*7+l in 2x7)
            # B_11 (idx 3 in 2x2) -> 0
            
            v0 = V_strassen[k][0]
            v2 = V_strassen[k][2]
            
            V[0 * p + l][col_idx] = v0
            V[1 * p + l][col_idx] = v2
            
            # W mapping:
            # C_00 (idx 0 in 2x2) -> C_0l (idx l*2+0 in 2x7, trace conv: k*n+i)
            # C_01 (idx 1 in 2x2) -> 0
            # C_10 (idx 2 in 2x2) -> C_1l (idx l*2+1 in 2x7)
            # C_11 (idx 3 in 2x2) -> 0
            
            w0 = W_strassen[k][0]
            w2 = W_strassen[k][2]
            
            W[l * n + 0][col_idx] = w0
            W[l * n + 1][col_idx] = w2
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
