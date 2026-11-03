import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 8
    # We will build U, V, W as lists of lists.
    # U: (n*m, rank) -> (12, rank)
    # V: (m*p, rank) -> (32, rank)
    # W: (p*n, rank) -> (24, rank)
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        # u_vec: length n*m
        # v_vec: length m*p
        # w_vec: length p*n
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    # Strassen decomposition for 2x2x2
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A B
    # Indices in A: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    # Indices in B: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    # Indices in C: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    
    def strassen_terms(a_idx, b_idx, c_idx):
        # a_idx: list of 4 indices into U (flattened)
        # b_idx: list of 4 indices into V (flattened)
        # c_idx: list of 4 indices into W (flattened)
        
        # M1 = (a11+a22)(b11+b22)
        u1 = [0]*(n*m); u1[a_idx[0]] = 1; u1[a_idx[3]] = 1
        v1 = [0]*(m*p); v1[b_idx[0]] = 1; v1[b_idx[3]] = 1
        w1 = [0]*(p*n)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # We need to accumulate W contributions.
        # Since we are adding terms one by one, we just add the W vector for this M.
        # But W depends on the combination.
        # It is easier to define the 7 M terms and then sum them in W.
        # However, the interface requires U, V, W columns.
        # Each column corresponds to one multiplication M.
        # So we add 7 columns.
        
        # M1
        w1[c_idx[0]] += 1
        w1[c_idx[3]] += 1
        add_term(u1, v1, w1)

        # M2 = (a21+a22)b11
        u2 = [0]*(n*m); u2[a_idx[2]] = 1; u2[a_idx[3]] = 1
        v2 = [0]*(m*p); v2[b_idx[0]] = 1
        w2 = [0]*(p*n)
        w2[c_idx[2]] += 1
        w2[c_idx[3]] -= 1
        add_term(u2, v2, w2)

        # M3 = a11(b12-b22)
        u3 = [0]*(n*m); u3[a_idx[0]] = 1
        v3 = [0]*(m*p); v3[b_idx[1]] = 1; v3[b_idx[3]] = -1
        w3 = [0]*(p*n)
        w3[c_idx[1]] += 1
        w3[c_idx[3]] += 1
        add_term(u3, v3, w3)

        # M4 = a22(b21-b11)
        u4 = [0]*(n*m); u4[a_idx[3]] = 1
        v4 = [0]*(m*p); v4[b_idx[2]] = 1; v4[b_idx[0]] = -1
        w4 = [0]*(p*n)
        w4[c_idx[0]] += 1
        w4[c_idx[2]] += 1
        add_term(u4, v4, w4)

        # M5 = (a11+a12)b22
        u5 = [0]*(n*m); u5[a_idx[0]] = 1; u5[a_idx[1]] = 1
        v5 = [0]*(m*p); v5[b_idx[3]] = 1
        w5 = [0]*(p*n)
        w5[c_idx[0]] -= 1
        w5[c_idx[1]] += 1
        add_term(u5, v5, w5)

        # M6 = (a21-a11)(b11+b12)
        u6 = [0]*(n*m); u6[a_idx[2]] = 1; u6[a_idx[0]] = -1
        v6 = [0]*(m*p); v6[b_idx[0]] = 1; v6[b_idx[1]] = 1
        w6 = [0]*(p*n)
        w6[c_idx[3]] += 1
        add_term(u6, v6, w6)

        # M7 = (a12-a22)(b21+b22)
        u7 = [0]*(n*m); u7[a_idx[1]] = 1; u7[a_idx[3]] = -1
        v7 = [0]*(m*p); v7[b_idx[2]] = 1; v7[b_idx[3]] = 1
        w7 = [0]*(p*n)
        w7[c_idx[0]] += 1
        add_term(u7, v7, w7)

    # Index helpers
    # A is 3x4. Flattened row-major.
    # A[i, j] -> i*4 + j
    # B is 4x8. Flattened row-major.
    # B[i, j] -> i*8 + j
    # C is 3x8. W[k*n+i, r] -> k*3 + i.
    # C[i, k] -> W[k*3 + i]
    
    # Block 1: A_top (rows 0,1) * B_L (cols 0,1,2,3)
    # A_top is 2x4.
    # Split A_top into A_TL (cols 0,1) and A_TR (cols 2,3).
    # A_TL indices: [0,1, 4,5] (0*4+0, 0*4+1, 1*4+0, 1*4+1)
    # A_TR indices: [2,3, 6,7]
    
    # B_L is 4x4.
    # Split B_L into B_LL1 (rows 0,1 cols 0,1), B_LL2 (rows 0,1 cols 2,3)
    # B_LR1 (rows 2,3 cols 0,1), B_LR2 (rows 2,3 cols 2,3)
    # B_LL1 indices: [0*8+0, 0*8+1, 1*8+0, 1*8+1] -> [0,1,8,9]
    # B_LL2 indices: [0*8+2, 0*8+3, 1*8+2, 1*8+3] -> [2,3,10,11]
    # B_LR1 indices: [2*8+0, 2*8+1, 3*8+0, 3*8+1] -> [16,17,24,25]
    # B_LR2 indices: [2*8+2, 2*8+3, 3*8+2, 3*8+3] -> [18,19,26,27]
    
    # C_TL1 (rows 0,1 cols 0,1) -> W indices:
    # C[0,0]->W[0], C[0,1]->W[1], C[1,0]->W[3], C[1,1]->W[4]
    # Wait, W[k*n+i].
    # C[i,k].
    # C[0,0] -> k=0, i=0 -> W[0]
    # C[0,1] -> k=1, i=0 -> W[1]
    # C[1,0] -> k=0, i=1 -> W[3]
    # C[1,1] -> k=1, i=1 -> W[4]
    # So c_idx = [0, 1, 3, 4]
    
    # C_TL2 (rows 0,1 cols 2,3)
    # C[0,2] -> W[2]
    # C[0,3] -> W[5]
    # C[1,2] -> W[6]
    # C[1,3] -> W[7]
    # c_idx = [2, 5, 6, 7]
    
    # C_TR1 (rows 2,3 cols 0,1) - Wait, A_top is only rows 0,1.
    # A_top corresponds to C rows 0,1.
    # So we only compute C rows 0,1 for the top part.
    
    # Let's refine the blocks.
    # A = [A_top; A_bot]
    # A_top: rows 0,1.
    # A_bot: row 2.
    
    # B = [B_L, B_R]
    # B_L: cols 0-3.
    # B_R: cols 4-7.
    
    # Part 1: A_top * B_L (2x4 * 4x4 -> 2x4)
    # C_TL (rows 0,1, cols 0,1)
    # C_TR (rows 0,1, cols 2,3)
    
    # C_TL = A_TL * B_LL1 + A_TR * B_LR1
    # C_TR = A_TL * B_LL2 + A_TR * B_LR2
    
    # Strassen for A_TL * B_LL1
    strassen_terms([0,1,4,5], [0,1,8,9], [0,1,3,4])
    # Strassen for A_TR * B_LR1
    strassen_terms([2,3,6,7], [16,17,24,25], [0,1,3,4])
    
    # Strassen for A_TL * B_LL2
    strassen_terms([0,1,4,5], [2,3,10,11], [2,5,6,7])
    # Strassen for A_TR * B_LR2
    strassen_terms([2,3,6,7], [18,19,26,27], [2,5,6,7])
    
    # Part 2: A_top * B_R (2x4 * 4x4 -> 2x4)
    # B_R cols 4-7.
    # B_RL1 (rows 0,1 cols 4,5)
    # B_RL2 (rows 0,1 cols 6,7)
    # B_RR1 (rows 2,3 cols 4,5)
    # B_RR2 (rows 2,3 cols 6,7)
    
    # B_RL1 indices: [0*8+4, 0*8+5, 1*8+4, 1*8+5] -> [4,5,12,13]
    # B_RL2 indices: [0*8+6, 0*8+7, 1*8+6, 1*8+7] -> [6,7,14,15]
    # B_RR1 indices: [2*8+4, 2*8+5, 3*8+4, 3*8+5] -> [20,21,28,29]
    # B_RR2 indices: [2*8+6, 2*8+7, 3*8+6, 3*8+7] -> [22,23,30,31]
    
    # C_TL_R (rows 0,1 cols 4,5)
    # C[0,4] -> W[4]
    # C[0,5] -> W[7]
    # C[1,4] -> W[9]
    # C[1,5] -> W[10]
    # c_idx = [4, 7, 9, 10]
    
    # C_TR_R (rows 0,1 cols 6,7)
    # C[0,6] -> W[6]
    # C[0,7] -> W[11]
    # C[1,6] -> W[12]
    # C[1,7] -> W[13]
    # c_idx = [6, 11, 12, 13]
    
    # Strassen for A_TL * B_RL1
    strassen_terms([0,1,4,5], [4,5,12,13], [4,7,9,10])
    # Strassen for A_TR * B_RR1
    strassen_terms([2,3,6,7], [20,21,28,29], [4,7,9,10])
    
    # Strassen for A_TL * B_RL2
    strassen_terms([0,1,4,5], [6,7,14,15], [6,11,12,13])
    # Strassen for A_TR * B_RR2
    strassen_terms([2,3,6,7], [22,23,30,31], [6,11,12,13])
    
    # Part 3: A_bot * B (1x4 * 4x8 -> 1x8)
    # A_bot: row 2. Indices [8,9,10,11].
    # B_L: cols 0-3.
    # B_R: cols 4-7.
    
    # C_Bot_L (row 2, cols 0-3)
    # C[2,0] -> W[0*3+2] = W[2]
    # C[2,1] -> W[1*3+2] = W[5]
    # C[2,2] -> W[2*3+2] = W[8]
    # C[2,3] -> W[3*3+2] = W[11]
    
    # C_Bot_R (row 2, cols 4-7)
    # C[2,4] -> W[4*3+2] = W[14]
    # C[2,5] -> W[5*3+2] = W[17]
    # C[2,6] -> W[6*3+2] = W[20]
    # C[2,7] -> W[7*3+2] = W[23]
    
    # A_bot * B_L
    # a1 B_LL1 + a2 B_LR1 + a3 B_LL2 + a4 B_LR2
    # a1 is A[2,0] -> idx 8
    # a2 is A[2,1] -> idx 9
    # a3 is A[2,2] -> idx 10
    # a4 is A[2,3] -> idx 11
    
    # Term a1 B_LL1
    # B_LL1 is 2x2.
    # C[2,0] = a1 * (B[0,0] + B[1,0])? No.
    # C[2,j] = sum_k A[2,k] B[k,j].
    # For j=0: A[2,0]B[0,0] + A[2,1]B[1,0] + A[2,2]B[2,0] + A[2,3]B[3,0].
    # We split B into blocks.
    # B_LL1 is B[0:2, 0:2].
    # Contribution to C[2,0] from B_LL1: A[2,0]B[0,0] + A[2,1]B[1,0].
    # This is not a single multiplication.
    # We need to decompose $A_{bot} B$ into rank 1 terms.
    # $C_{bot, j} = \sum_{k=0}^3 A_{2,k} B_{k,j}$.
    # Each term $A_{2,k} B_{k,j}$ is a scalar product.
    # So we add 4 terms for each column j?
    # No, $B_{k,j}$ is a scalar.
    # So for each $k, j$, we have a term $A_{2,k} B_{k,j} E_{2,j}$.
    # Total 16 terms.
    
    for k in range(4):
        a_idx = 2*4 + k
        for j in range(8):
            b_idx = k*8 + j
            w_idx = j*3 + 2
            u = [0]*(n*m); u[a_idx] = 1
            v = [0]*(m*p); v[b_idx] = 1
            w = [0]*(p*n); w[w_idx] = 1
            add_term(u, v, w)

    # Convert to matrices
    rank = len(U)
    U_mat = [row[:] for row in zip(*U)] if U else []
    V_mat = [row[:] for row in zip(*V)] if V else []
    W_mat = [row[:] for row in zip(*W)] if W else []
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
