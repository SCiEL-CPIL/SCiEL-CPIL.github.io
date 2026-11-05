"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We decompose B (4x5) into B1 (4x4) and B2 (4x1)
    # C = A * B = [A*B1 | A*B2] where A*B1 is 4x4 and A*B2 is 4x1
    # Use Strassen (rank 49) for 4x4x4 part, schoolbook (rank 16) for 4x4x1 part
    # Total rank = 49 + 16 = 65
    
    # Strassen 2x2x2 decomposition
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, 0, 0]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    # A is 4x4, split into 2x2 blocks: A11(top-left), A12(top-right), A21(bottom-left), A22(bottom-right)
    # B is 4x4, same split
    # C is 4x4, same split
    
    # For the 4x4x4 part, we use 7x7 = 49 Strassen multiplications
    # Each "multiplication" in the Strassen sense corresponds to one column
    
    rank_strassen = 49
    U_strassen = [[0] * rank_strassen for _ in range(16)]  # 4*4 = 16
    V_strassen = [[0] * rank_strassen for _ in range(16)]  # 4*4 = 16
    W_strassen = [[0] * rank_strassen for _ in range(16)]  # 4*4 = 16
    
    # Block structure for 4x4 matrices (2x2 blocks of 2x2)
    # Block (bi, bj) where bi, bj in {0,1}
    # Within block, row r in {0,1}, col c in {0,1}
    # Global row = bi*2 + r, global col = bj*2 + c
    # Linear index in U (A): (bi*2+r)*4 + (bj*2+c)
    # Linear index in V (B): (bj*2+r)*4 + (bk*2+c) -- wait, B is m x p = 4 x 4
    # Actually for A*B where both are 4x4:
    # A is n x m = 4 x 4, B is m x p = 4 x 4, C is n x p = 4 x 4
    # U index: i*m + j = i*4 + j
    # V index: j*p + k = j*4 + k  
    # W index: k*n + i = k*4 + i
    
    # For Strassen tiling:
    # A blocks: A[bi][bj] is 2x2, bi,bj in {0,1}
    # B blocks: B[bj][bk] is 2x2, bj,bk in {0,1}
    # C blocks: C[bi][bk] is 2x2, bi,bk in {0,1}
    
    # The 49 Strassen multiplications correspond to (s1, s2) where s1,s2 in {0..6}
    # For each pair of blocks (bi,bj) and (bj,bk), we use the Strassen decomposition
    
    # Actually, the tiling works as follows:
    # C = sum over bj of A_block(bi,bj) * B_block(bj,bk)
    # Each A_block * B_block is a 2x2 * 2x2 multiplication using Strassen (7 mults)
    # But we need to combine across bj
    
    # Hmm, this is getting complicated. The standard Strassen tiling for 4x4x4 
    # gives rank 49 because we have 4 block columns and use 7 Strassen mults per pair.
    
    # Let me use a different, cleaner approach: just build the Kronecker product structure.
    
    # For 4x4x4 using Strassen: rank = 7^2 = 49
    # U4[i*4+j, s1*7+s2] = U2[i_block*2+i_inner, s1] * U2[j_block*2+j_inner, s2]
    # where i = i_block*2 + i_inner, j = j_block*2 + j_inner
    
    # Let me redo this more carefully.
    
    # Clear and rebuild
    U_strassen = [[0] * 49 for _ in range(16)]
    V_strassen = [[0] * 49 for _ in range(16)]
    W_strassen = [[0] * 49 for _ in range(16)]
    
    for i in range(4):
        for j in range(4):
            ib = i // 2  # block row of A
            ic = i % 2   # inner row
            jb = j // 2  # block col of A
            jc = j % 2   # inner col
            for k in range(4):
                kb = k // 2  # block row of B (which is col block of A)
                kc = k % 2   # inner col of B
                # For V: index is j*4 + k
                jvb = j // 2
                jvc = j % 2
                # V index: j*4 + k where j is row of B, k is col of B
                # B is m x p = 4 x 4
                # V index = j * p + k = j * 4 + k
                for s1 in range(7):
                    for s2 in range(7):
                        col = s1 * 7 + s2
                        # U contribution: U2[ib*2+ic, s1] * U2[jb*2+jc, s2]
                        u_val = U2[ib*2+ic][s1] * U2[jb*2+jc][s2]
                        U_strassen[i*4+j][col] += u_val
                        
                        # V contribution: V2[jb*2+jvc, s1] * V2[kb*2+kc, s2]  
                        # V index is j*4+k, so row is j, col is k
                        v_row = j * 4 + k
                        v_val = V2[jb*2+jvc][s1] * V2[kb*2+kc][s2]
                        V_strassen[v_row][col] += v_val
                        
                        # W contribution: W2[ib*2+ic, s1] * W2[kb*2+kc, s2]
                        # W index is k*4 + i
                        w_row = k * 4 + i
                        w_val = W2[ib*2+ic][s1] * W2[kb*2+kc][s2]
                        W_strassen[w_row][col] += w_val
    
    # Now build the 4x4x1 schoolbook part (rank 16)
    rank_school = 16
    U_school = [[0] * rank_school for _ in range(16)]
    V_school = [[0] * rank_school for _ in range(4)]   # m*p = 4*1 = 4
    W_school = [[0] * rank_school for _ in range(4)]   # p*n = 1*4 = 4
    
    col = 0
    for i in range(4):
        for j in range(4):
            U_school[i*4 + j][col] = 1
            V_school[j * 1 + 0][col] = 1  # V index: j*p + k = j*1 + 0
            W_school[0 * 4 + i][col] = 1   # W index: k*n + i = 0*4 + i
            col += 1
    
    # Combine: total rank = 49 + 16 = 65
    total_rank = 49 + 16
    
    # U: (16, 65)
    U = [[0] * total_rank for _ in range(16)]
    for i in range(16):
        for c in range(49):
            U[i][c] = U_strassen[i][c]
        for c in range(16):
            U[i][49 + c] = U_school[i][c]
    
    # V: (20, 65) -- m*p = 4*5 = 20
    V = [[0] * total_rank for _ in range(20)]
    for i in range(16):
        for c in range(49):
            V[i][c] = V_strassen[i][c]
    for i in range(4):
        for c in range(16):
            V[16 + i][49 + c] = V_school[i][c]
    
    # W: (20, 65) -- p*n = 5*4 = 20
    W = [[0] * total_rank for _ in range(20)]
    for i in range(16):
        for c in range(49):
            W[i][c] = W_strassen[i][c]
    for i in range(4):
        for c in range(16):
            W[16 + i][49 + c] = W_school[i][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
