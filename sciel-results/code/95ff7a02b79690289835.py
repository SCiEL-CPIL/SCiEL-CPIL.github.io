"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1; W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1; V[0][1] = 1; W[2][1] = 1; W[3][1] = -1
        U[0][2] = 1; V[1][2] = 1; V[3][2] = -1; W[1][2] = 1; W[3][2] = 1
        U[3][3] = 1; V[2][3] = 1; V[0][3] = -1; W[0][3] = 1; W[2][3] = 1
        U[0][4] = 1; U[1][4] = 1; V[3][4] = 1; W[1][4] = 1; W[0][4] = -1
        U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1; V[2][6] = 1; V[3][6] = 1; W[0][6] = 1
        
        return U, V, W
    
    sU, sV, sW = strassen_2x2x2()
    
    U_444 = [[0]*49 for _ in range(16)]
    V_444 = [[0]*49 for _ in range(16)]
    W_444 = [[0]*49 for _ in range(16)]
    
    def add_strassen_mult_v2(t, a_blocks, b_blocks, c_blocks_with_signs):
        col_base = t * 7
        for c in range(7):
            col = col_base + c
            for u_idx in range(4):
                val = sU[u_idx][c]
                if val != 0:
                    for _, block_offset in a_blocks:
                        U_444[block_offset + u_idx][col] += val
            for v_idx in range(4):
                val = sV[v_idx][c]
                if val != 0:
                    for _, block_offset in b_blocks:
                        V_444[block_offset + v_idx][col] += val
            for w_idx in range(4):
                val = sW[w_idx][c]
                if val != 0:
                    for block_offset, sign in c_blocks_with_signs:
                        W_444[block_offset + w_idx][col] += sign * val
    
    add_strassen_mult_v2(0, [(0,0),(3,12)], [(0,0),(3,12)], [(0,1),(12,1)])
    add_strassen_mult_v2(1, [(2,8),(3,12)], [(0,0)], [(8,1),(12,-1)])
    add_strassen_mult_v2(2, [(0,0)], [(1,4),(3,12)], [(4,1),(12,1)])
    add_strassen_mult_v2(3, [(3,12)], [(2,8),(0,0)], [(0,1),(8,1)])
    add_strassen_mult_v2(4, [(0,0),(1,4)], [(3,12)], [(4,1),(0,-1)])
    add_strassen_mult_v2(5, [(2,8),(0,0)], [(0,0),(1,4)], [(12,1)])
    add_strassen_mult_v2(6, [(1,4),(3,12)], [(2,8),(3,12)], [(0,1)])
    
    rank_total = 49 + 48
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(28)]
    W = [[0]*rank_total for _ in range(28)]
    
    for i in range(16):
        for c in range(49):
            U[i][c] = U_444[i][c]
    for j in range(16):
        for c in range(49):
            V[j][c] = V_444[j][c]
    for k in range(16):
        for c in range(49):
            W[k][c] = W_444[k][c]
    
    col = 49
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i*4 + j][col] = 1
                V[j*7 + 4 + k][col] = 1
                W[(4+k)*4 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
