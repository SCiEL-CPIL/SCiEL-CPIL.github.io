import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # Part 2: 4x4x3 Strassen (Rank 49)
    
    # We will build the factors manually
    
    rank_444 = 49
    rank_443 = 49
    total_rank = rank_444 + rank_443
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Helper to get Strassen 2x2 factors
    # Returns U2, V2, W2 of shape (4, 7)
    def get_strassen_2x2():
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # Indices: 0:00, 1:01, 2:10, 3:11
        # Term 1: (A00+A11)(B00+B11) -> C00+C11
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11
        
        # Term 2: (A10+A11)B00 -> C10
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1 # C10
        
        # Term 3: A00(B01-B11) -> C01
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[1, 2] = 1 # C01
        
        # Term 4: A11(B10-B00) -> C00, C10
        U2[3, 3] = 1
        V2[2, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = 1; W2[2, 3] = 1 # C00, C10
        
        # Term 5: (A00+A01)B11 -> C00, C01
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[1, 4] = 1 # C00, C01 (Note: C00 coeff is -1 in formula C00=P1+P4-P5+P7)
        
        # Term 6: (A10-A00)(B00+B01) -> C11
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1 # C11
        
        # Term 7: (A01-A11)(B10+B11) -> C00, C11
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1; W2[3, 6] = -1 # C00, C11 (C00 coeff +1, C11 coeff -1 in formula C11=P1-P2+P3+P6? No, C11=P1-P2+P3+P6. Wait.)
        
        # Let's re-verify the C formulas.
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # Term 1 (P1): C00+1, C11+1. W2[0,0]=1, W2[3,0]=1. Correct.
        # Term 2 (P2): C10+1, C11-1. W2[2,1]=1, W2[3,1]=-1.
        #   My code: W2[2,1]=1. Missing W2[3,1]=-1.
        # Term 3 (P3): C01+1, C11+1. W2[1,2]=1, W2[3,2]=1.
        #   My code: W2[1,2]=1. Missing W2[3,2]=1.
        # Term 4 (P4): C00+1, C10+1. W2[0,3]=1, W2[2,3]=1. Correct.
        # Term 5 (P5): C00-1, C01+1. W2[0,4]=-1, W2[1,4]=1. Correct.
        # Term 6 (P6): C11+1. W2[3,5]=1. Correct.
        # Term 7 (P7): C00+1, C11-1. W2[0,6]=1, W2[3,6]=-1. Correct.
        
        # Fix W2
        W2[3, 1] = -1
        W2[3, 2] = 1
        
        return U2, V2, W2

    U2, V2, W2 = get_strassen_2x2()
    
    # Build 4x4x4 factors
    # 7 Strassen terms for 4x4
    # Each term t has a linear form on A blocks and B blocks
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    # Coefficients for A blocks in each of the 7 terms
    # t=0: A00+A11
    # t=1: A10+A11
    # t=2: A00
    # t=3: A11
    # t=4: A00+A01
    # t=5: A10-A00
    # t=6: A01-A11
    
    A_coeffs = [
        {0: 1, 3: 1}, # t=0
        {2: 1, 3: 1}, # t=1
        {0: 1},       # t=2
        {3: 1},       # t=3
        {0: 1, 1: 1}, # t=4
        {2: 1, 0: -1},# t=5
        {1: 1, 3: -1} # t=6
    ]
    
    # Coefficients for B blocks
    B_coeffs = [
        {0: 1, 3: 1}, # t=0: B00+B11
        {0: 1},       # t=1: B00
        {1: 1, 3: -1},# t=2: B01-B11
        {2: 1, 0: -1},# t=3: B10-B00
        {3: 1},       # t=4: B11
        {0: 1, 1: 1}, # t=5: B00+B01
        {2: 1, 3: 1}  # t=6: B10+B11
    ]
    
    # Coefficients for C blocks in the final sum?
    # No, the 7 terms P_t are intermediate.
    # The final C is a linear combination of P_t.
    # But in the tensor decomposition, we just sum the 7 terms.
    # The W2 matrix already accounts for the C coefficients within each 2x2 block.
    # So we just need to map the 2x2 block indices to 4x4 indices.
    
    col_offset = 0
    for t in range(7):
        for s in range(7):
            # U: A part
            for block_idx, coeff_a in A_coeffs[t].items():
                # block_idx 0:00, 1:01, 2:10, 3:11
                # i = block_idx // 2, j = block_idx % 2
                i = block_idx // 2
                j = block_idx % 2
                # U2 indices for block (i,j)
                # U2[0] is A00, U2[1] is A01, U2[2] is A10, U2[3] is A11
                # But U2 is for the 2x2 matrix A_t.
                # The entries of A_t are A_{2i+k, 2j+l}
                # U2[k*2+l, s] is the coefficient for A_t[k,l]
                for k in range(2):
                    for l in range(2):
                        u_idx = (2*i + k) * 4 + (2*j + l)
                        U[u_idx, col_offset] += coeff_a * U2[k*2+l, s]
                        
            # V: B part
            for block_idx, coeff_b in B_coeffs[t].items():
                i = block_idx // 2
                j = block_idx % 2
                for k in range(2):
                    for l in range(2):
                        v_idx = (2*i + k) * 7 + (2*j + l)
                        V[v_idx, col_offset] += coeff_b * V2[k*2+l, s]
                        
            # W: C part
            # The output C_t is a 2x2 matrix.
            # It contributes to C blocks.
            # Which C blocks?
            # The 7 Strassen terms P_t for 4x4 correspond to specific C blocks.
            # P1 -> C00, C01, C10, C11? No.
            # The 4x4 Strassen algorithm computes C00, C01, C10, C11 from P1..P7.
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P4
            # C11 = P1 - P2 + P3 + P6
            
            # So for term t, we need to know which C blocks it contributes to.
            # And with what coefficient.
            
            C_contrib = {
                0: {0: 1, 3: 1}, # t=0 (P1): C00+1, C11+1
                1: {2: 1, 3: -1},# t=1 (P2): C10+1, C11-1
                2: {1: 1, 3: 1}, # t=2 (P3): C01+1, C11+1
                3: {0: 1, 2: 1}, # t=3 (P4): C00+1, C10+1
                4: {0: -1, 1: 1},# t=4 (P5): C00-1, C01+1
                5: {3: 1},       # t=5 (P6): C11+1
                6: {0: 1, 3: -1} # t=6 (P7): C00+1, C11-1
            }
            
            for c_block_idx, coeff_c in C_contrib[t].items():
                i = c_block_idx // 2
                j = c_block_idx % 2
                for k in range(2):
                    for l in range(2):
                        w_idx = (2*j + l) * 4 + (2*i + k)
                        W[w_idx, col_offset] += coeff_c * W2[k*2+l, s]
                        
            col_offset += 1
            
    # Part 2: 4x4x3
    # Same as 4x4x4 but B has only 3 columns.
    # We can just copy the U and W parts, and truncate V.
    # V for 4x4x4 has indices (i, j) where i in 0..3, j in 0..3.
    # V for 4x4x3 has indices (i, j) where i in 0..3, j in 0..2.
    # So we just take the first 3 columns of each 2x2 block?
    # No, B is 4x3.
    # The blocks are 2x1.5? No.
    # We can just use the same decomposition and ignore the terms involving B[:, 3].
    # But the rank is still 49.
    
    # We will construct V2_443 by taking V from 444 and removing the 4th column of B.
    # V_444 shape (28, 49).
    # V_443 shape (21, 49).
    # B indices: j (0..3) * 4 + k (0..3).
    # For 4x3: j (0..3) * 3 + k (0..2).
    
    # It's easier to just generate it separately or copy.
    # Since the structure is identical, we can copy U and W.
    # For V, we map the indices.
    
    U_443 = U[:, :49].copy()
    W_443 = W[:, :49].copy()
    
    V_443 = np.zeros((21, 49))
    for j in range(4):
        for k in range(3):
            v_idx_443 = j * 3 + k
            v_idx_444 = j * 4 + k
            V_443[v_idx_443, :] = V[v_idx_444, :49]
            
    U[:, 49:] = U_443
    V[:, 49:] = V_443
    W[:, 49:] = W_443
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
