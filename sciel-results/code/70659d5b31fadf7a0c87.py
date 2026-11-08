import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 98: Two independent rank-49 4x4 multiplications
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We will fill the first 49 columns for k in 0..3
    # and the next 49 columns for k in 4..7
    
    # Helper to get Strassen 2x2 factors
    # Returns U2, V2, W2 of shape (4, 7)
    # U2[i*2+j, s], V2[j*2+k, s], W2[k*2+i, s]
    def get_strassen_2x2():
        U2 = [[0]*7 for _ in range(4)]
        V2 = [[0]*7 for _ in range(4)]
        W2 = [[0]*7 for _ in range(4)]
        
        # M1
        U2[0][0] = 1; U2[3][0] = 1
        V2[0][0] = 1; V2[3][0] = 1
        W2[0][0] = 1; W2[3][0] = 1
        
        # M2
        U2[2][1] = 1; U2[3][1] = 1
        V2[0][1] = 1
        W2[1][1] = 1 # C10 -> i=1, k=0 -> row 1
        
        # M3
        U2[0][2] = 1
        V2[2][2] = 1; V2[3][2] = -1
        W2[2][2] = 1 # C01 -> i=0, k=1 -> row 2
        
        # M4
        U2[3][3] = 1
        V2[1][3] = 1; V2[0][3] = -1
        W2[0][3] = 1; W2[1][3] = 1 # C00, C10
        
        # M5
        U2[0][4] = 1; U2[1][4] = 1
        V2[3][4] = 1
        W2[0][4] = -1; W2[2][4] = 1 # C00, C01
        
        # M6
        U2[2][5] = 1; U2[0][5] = -1
        V2[0][5] = 1; V2[2][5] = 1
        W2[3][5] = 1 # C11 -> i=1, k=1 -> row 3
        
        # M7
        U2[1][6] = 1; U2[3][6] = -1
        V2[3][6] = 1; V2[1][6] = 1
        W2[0][6] = 1; W2[2][6] = 1 # C00, C01
        
        return U2, V2, W2

    U2, V2, W2 = get_strassen_2x2()
    
    # Strassen 4x4 coefficients
    # A_blocks: 00, 01, 10, 11
    # B_blocks: 00, 01, 10, 11
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B10)
    # M7: (A01-A11)(B11+B01)
    
    # Coefficients for A blocks in each M
    # alpha[r][block]
    alpha = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # Coefficients for B blocks in each M
    beta = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [0, 1, 0, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 0, 1, 0], # M6
        [0, 1, 0, 1]  # M7
    ]
    
    # Coefficients for C blocks from M
    # gamma[block][r]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    gamma = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Function to fill one 4x4 product
    # offset_col: starting column index in U, V, W
    # k_offset: starting k index (0 or 4)
    def fill_4x4(offset_col, k_offset):
        for r in range(7):
            # For each Strassen term r, we decompose the 2x2 matrix product
            # A_eff * B_eff
            # A_eff is a 4x4 matrix where block (a,b) is alpha[r][block] * A_block
            # B_eff is a 4x4 matrix where block (c,d) is beta[r][block] * B_block
            
            # We use the 2x2 Strassen factors U2, V2, W2
            # The "scalar" multiplication is done on the 2x2 blocks
            # So for each of the 7 terms in the 2x2 decomposition (s=0..6):
            for s in range(7):
                col = offset_col + r * 7 + s
                
                # Update U
                # U2[s] gives coefficients for the 2x2 "A" matrix
                # The "A" matrix here is A_eff
                # A_eff block (a,b) = alpha[r][ab] * A_ab
                # U2[i*2+j, s] is the coefficient for A_{i,j} in the 2x2 decomposition
                # Here A_{i,j} corresponds to the scalar entry in the 2x2 block matrix
                # So the contribution to the 4x4 A matrix is:
                # For block (a,b) of A, the scalar entry (u,v) gets coefficient:
                # alpha[r][a*2+b] * U2[u*2+v, s]
                
                # Let's iterate over blocks of A
                for a in range(2):
                    for b in range(2):
                        block_idx = a * 2 + b
                        coeff_a = alpha[r][block_idx]
                        if coeff_a == 0: continue
                        
                        # For each scalar entry in this 2x2 block
                        for u in range(2):
                            for v in range(2):
                                # A row: a*2+u, col: b*2+v
                                # U2 index: u*2+v
                                coeff_u2 = U2[u*2+v][s]
                                if coeff_u2 == 0: continue
                                
                                row_U = (a*2+u) * m + (b*2+v)
                                U[row_U][col] += coeff_a * coeff_u2
                                
                # Update V
                for c in range(2):
                    for d in range(2):
                        block_idx = c * 2 + d
                        coeff_b = beta[r][block_idx]
                        if coeff_b == 0: continue
                        
                        for u in range(2):
                            for v in range(2):
                                # B row: c*2+u, col: d*2+v
                                # V2 index: u*2+v
                                coeff_v2 = V2[u*2+v][s]
                                if coeff_v2 == 0: continue
                                
                                row_V = (c*2+u) * p + (d*2+v + k_offset)
                                V[row_V][col] += coeff_b * coeff_v2
                                
                # Update W
                # W2[k*2+i, s] is the coefficient for C_{k,i} in the 2x2 decomposition
                # The "C" matrix here is C_eff
                # C_eff block (k,i) is the result of the 2x2 multiplication
                # The total C block (k,i) is sum_r gamma[k*2+i][r] * C_eff_block(k,i)
                
                for k_block in range(2):
                    for i_block in range(2):
                        block_idx = k_block * 2 + i_block
                        coeff_g = gamma[block_idx][r]
                        if coeff_g == 0: continue
                        
                        for u in range(2):
                            for v in range(2):
                                # C row: i_block*2+u, col: k_block*2+v
                                # W2 index: u*2+v (since W2 is k*2+i? No, W2[k*2+i])
                                # In get_strassen_2x2, W2 row is k*2+i.
                                # So W2[u*2+v] corresponds to C_{v,u}?
                                # Let's check get_strassen_2x2.
                                # W2[0] -> C00 (i=0, k=0).
                                # W2[1] -> C10 (i=1, k=0).
                                # W2[2] -> C01 (i=0, k=1).
                                # W2[3] -> C11 (i=1, k=1).
                                # So W2[k*2+i].
                                # Here u corresponds to k? No.
                                # In V2, index is j*2+k.
                                # In U2, index is i*2+j.
                                # In W2, index is k*2+i.
                                
                                # So for C block (k_block, i_block), the scalar entry (v, u)
                                # corresponds to C_{i_block*2+u, k_block*2+v}
                                # The W2 index for this scalar is (k_block*2+v)*2 + (i_block*2+u)?
                                # No, W2 is for the 2x2 block matrix.
                                # The 2x2 block matrix C_block has entries C_{kl} where k,l in 0..1.
                                # W2[k*2+l] is the coefficient for C_{k,l} in the 2x2 Strassen.
                                # So for block (k_block, i_block), the scalar (v, u) is at
                                # position (v, u) in the 2x2 block matrix?
                                # No, the 2x2 block matrix has rows k_block, i_block?
                                # The 4x4 C matrix is partitioned into 2x2 blocks.
                                # Block (k_block, i_block) is the submatrix C_{i_block*2:..., k_block*2:...}
                                # The entries are C_{i_block*2+u, k_block*2+v}.
                                # In the 2x2 block matrix, this entry is at row i_block, col k_block?
                                # No, the block index is (row_block, col_block).
                                # The Strassen W2 gives coefficients for the 2x2 block matrix C_block.
                                # C_block[i_block, k_block] is the block.
                                # Wait, C is $n \times p$. $n=4, p=4$.
                                # Block (row, col).
                                # W2 index is k*2+i.
                                # This implies W2 is indexed by (col, row) of the 2x2 block matrix?
                                # Let's re-verify W2 in get_strassen_2x2.
                                # W2[0] -> C00. i=0, k=0.
                                # W2[1] -> C10. i=1, k=0.
                                # W2[2] -> C01. i=0, k=1.
                                # W2[3] -> C11. i=1, k=1.
                                # So W2[k*2+i].
                                
                                # So for the 2x2 block matrix, the entry at row i_block, col k_block
                                # has coefficient W2[k_block*2+i_block][s].
                                
                                coeff_w2 = W2[k_block*2+i_block][s]
                                if coeff_w2 == 0: continue
                                
                                # Total coefficient for C_{i_block*2+u, k_block*2+v}
                                # is coeff_g * coeff_w2
                                # Note: coeff_g is for the whole block?
                                # gamma[block_idx][r] is the coefficient for the block (k_block, i_block)
                                # in the sum of M_r.
                                # So yes, it applies to all entries in that block.
                                
                                row_W = (k_block*2+v + k_offset) * n + (i_block*2+u)
                                W[row_W][col] += coeff_g * coeff_w2
                                
    fill_4x4(0, 0)
    fill_4x4(49, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
