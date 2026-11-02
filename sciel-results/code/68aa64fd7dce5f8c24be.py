"""Matrix multiplication 2,2,8 - rank-28 exact tensor decomposition via 4x Strassen."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose M_{2,2,8} as direct sum of 4 copies of M_{2,2,2}.
    # Each copy uses Strassen's rank-7 algorithm. Total rank = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]   # 4 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 16 x 28
    
    # Strassen's 7 products for 2x2 * 2x2 -> 2x2
    # A indices: 00->0, 01->1, 10->2, 11->3
    # B indices: 00->0, 01->1, 10->2, 11->3
    # C indices: 00->0, 10->1, 01->2, 11->3  (C[i,j] = A[:,i] @ B[j,:], trace convention W[k*n+i,r])
    
    # U coefficients: for each Strassen product s, which A elements with what signs
    U_coeffs = [
        {0: 1, 3: 1},       # M1 = (A00 + A11)
        {2: 1, 3: 1},       # M2 = (A10 + A11)
        {0: 1},             # M3 = A00
        {3: 1},             # M4 = A11
        {0: 1, 1: 1},       # M5 = (A00 + A01)
        {2: 1, 0: -1},      # M6 = (A10 - A00)
        {1: 1, 3: -1},      # M7 = (A01 - A11)
    ]
    
    # V coefficients: for each Strassen product s, which B elements with what signs
    V_coeffs = [
        {0: 1, 3: 1},       # M1: (B00 + B11)
        {0: 1},             # M2: B00
        {1: 1, 3: -1},      # M3: (B01 - B11)
        {2: 1, 0: -1},      # M4: (B10 - B00)
        {3: 1},             # M5: B11
        {0: 1, 1: 1},       # M6: (B00 + B01)
        {2: 1, 3: 1},       # M7: (B10 + B11)
    ]
    
    # W coefficients: for each Strassen product s, which C elements with what signs
    # C local indices: 00->0, 10->1, 01->2, 11->3
    W_coeffs = [
        {0: 1, 3: 1},       # M1: C00 += M1, C11 += M1
        {1: 1, 3: -1},      # M2: C10 += M2, C11 -= M2
        {2: 1, 3: 1},       # M3: C01 += M3, C11 += M3
        {0: 1, 1: 1},       # M4: C00 += M4, C10 += M4
        {0: -1, 2: 1},      # M5: C00 -= M5, C01 += M5
        {3: 1},             # M6: C11 += M6
        {0: 1},             # M7: C00 += M7
    ]
    
    for q in range(4):
        # Block q handles columns 2*q, 2*q+1 of B and rows 2*q, 2*q+1 of C
        for s in range(7):
            r = q * 7 + s
            
            # Fill U (same for all blocks since A is shared)
            for a_idx, val in U_coeffs[s].items():
                U[a_idx][r] = val
            
            # Fill V
            # Local B index: 0->(j=0,k_local=0), 1->(j=0,k_local=1), 2->(j=1,k_local=0), 3->(j=1,k_local=1)
            for b_local, val in V_coeffs[s].items():
                j = b_local // 2
                k_local = b_local % 2
                k = 2 * q + k_local
                v_idx = j * p + k
                V[v_idx][r] = val
            
            # Fill W
            # Local C index: 0->(i=0,k_local=0), 1->(i=1,k_local=0), 2->(i=0,k_local=1), 3->(i=1,k_local=1)
            for c_local, val in W_coeffs[s].items():
                i = c_local % 2
                k_local = c_local // 2
                k = 2 * q + k_local
                w_idx = k * n + i
                W[w_idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
