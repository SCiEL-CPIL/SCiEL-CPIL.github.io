import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target rank 22
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a block
    # block_i_offset: offset in i (0 or 1)
    # block_j_offset: offset in j (0, 1, or 2)
    # block_k_offset: offset in k (0, 1, 2, or 3)
    # block_n, block_m, block_p: dimensions of the block
    # u_block, v_block, w_block: local matrices
    
    def add_block(i_off, j_off, k_off, bn, bm, bp, u_loc, v_loc, w_loc):
        nonlocal col
        for i in range(bn):
            for j in range(bm):
                for r in range(len(u_loc[0])):
                    if u_loc[i][j][r] != 0:
                        U[(i_off + i) * m + (j_off + j)][col + r] += u_loc[i][j][r]
        for j in range(bm):
            for k in range(bp):
                for r in range(len(v_loc[0])):
                    if v_loc[j][k][r] != 0:
                        V[(j_off + j) * p + (k_off + k)][col + r] += v_loc[j][k][r]
        for k in range(bp):
            for i in range(bn):
                for r in range(len(w_loc[0])):
                    if w_loc[k][i][r] != 0:
                        W[(k_off + k) * n + (i_off + i)][col + r] += w_loc[k][i][r]
        col += len(u_loc[0])
        
    # 1. Strassen 2x2x2 for j=1,2, k=1,2
    # Local indices 0,1
    u_s = [[[0]*7 for _ in range(2)] for _ in range(2)]
    v_s = [[[0]*7 for _ in range(2)] for _ in range(2)]
    w_s = [[[0]*7 for _ in range(2)] for _ in range(2)]
    
    # M1
    u_s[0][0][0] = 1; u_s[1][1][0] = 1
    v_s[0][0][0] = 1; v_s[1][1][0] = 1
    w_s[0][0][0] = 1; w_s[1][1][0] = 1
    
    # M2
    u_s[1][0][1] = 1; u_s[1][1][1] = 1
    v_s[0][0][1] = 1
    w_s[1][0][1] = 1; w_s[1][1][1] = -1
    
    # M3
    u_s[0][0][2] = 1
    v_s[0][1][2] = 1; v_s[1][1][2] = -1
    w_s[0][1][2] = 1; w_s[1][1][2] = 1
    
    # M4
    u_s[1][1][3] = 1
    v_s[1][0][3] = 1; v_s[0][0][3] = -1
    w_s[0][0][3] = 1; w_s[1][0][3] = 1
    
    # M5
    u_s[0][0][4] = 1; u_s[0][1][4] = 1
    v_s[1][1][4] = 1
    w_s[0][0][4] = -1; w_s[0][1][4] = 1
    
    # M6
    u_s[1][0][5] = 1; u_s[0][0][5] = -1
    v_s[0][0][5] = 1; v_s[0][1][5] = 1
    w_s[1][1][5] = 1
    
    # M7
    u_s[0][1][6] = 1; u_s[1][1][6] = -1
    v_s[1][0][6] = 1; v_s[1][1][6] = 1
    w_s[0][0][6] = 1
    
    add_block(0, 0, 0, 2, 2, 2, u_s, v_s, w_s)
    
    # 2. Schoolbook 2x1x2 for j=3, k=1,2
    # Rank 4
    u_sb = [[[0]*4 for _ in range(1)] for _ in range(2)]
    v_sb = [[[0]*4 for _ in range(2)] for _ in range(1)]
    w_sb = [[[0]*4 for _ in range(2)] for _ in range(2)]
    
    # C_ik = A_i1 B_1k
    # k=0,1 local
    for k_loc in range(2):
        for i_loc in range(2):
            r = i_loc * 2 + k_loc
            u_sb[i_loc][0][r] = 1
            v_sb[0][k_loc][r] = 1
            w_sb[k_loc][i_loc][r] = 1
            
    add_block(0, 2, 0, 2, 1, 2, u_sb, v_sb, w_sb)
    
    # 3. Strassen 2x2x2 for j=1,2, k=3,4
    add_block(0, 0, 2, 2, 2, 2, u_s, v_s, w_s)
    
    # 4. Schoolbook 2x1x2 for j=3, k=3,4
    add_block(0, 2, 2, 2, 1, 2, u_sb, v_sb, w_sb)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
