"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(u_idx, v_idx, w_idx, col):
        # u_idx: [u00, u01, u10, u11]
        # v_idx: [v00, v01, v10, v11]
        # w_idx: [w00, w01, w10, w11]
        
        # M1 = (u00 + u11)(v00 + v11)
        U[u_idx[0]][col] += 1
        U[u_idx[3]][col] += 1
        V[v_idx[0]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[0]][col] += 1
        W[w_idx[3]][col] += 1
        
        # M2 = (u10 + u11)v00
        U[u_idx[2]][col] += 1
        U[u_idx[3]][col] += 1
        V[v_idx[0]][col] += 1
        W[w_idx[2]][col] += 1
        W[w_idx[3]][col] += -1
        
        # M3 = u00(v01 - v11)
        U[u_idx[0]][col] += 1
        V[v_idx[1]][col] += 1
        V[v_idx[3]][col] += -1
        W[w_idx[1]][col] += 1
        W[w_idx[3]][col] += 1
        
        # M4 = u11(v10 - v00)
        U[u_idx[3]][col] += 1
        V[v_idx[2]][col] += 1
        V[v_idx[0]][col] += -1
        W[w_idx[0]][col] += 1
        W[w_idx[2]][col] += 1
        
        # M5 = (u00 + u01)v11
        U[u_idx[0]][col] += 1
        U[u_idx[1]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[0]][col] += -1
        W[w_idx[1]][col] += 1
        
        # M6 = (u10 - u00)(v00 + v01)
        U[u_idx[2]][col] += 1
        U[u_idx[0]][col] += -1
        V[v_idx[0]][col] += 1
        V[v_idx[1]][col] += 1
        W[w_idx[3]][col] += 1
        
        # M7 = (u01 - u11)(v10 + v11)
        U[u_idx[1]][col] += 1
        U[u_idx[3]][col] += -1
        V[v_idx[2]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[0]][col] += 1

    col = 0
    
    # Top 2 rows of A (i=0,1)
    # A_L: i in {0,1}, j in {0,1}
    # A_R: i in {0,1}, j in {2,3}
    
    # Block 1: A_L * B_L00 (B rows 0,1 cols 0,1) -> C cols 0,1
    u_idx = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # A[0,0], A[0,1], A[1,0], A[1,1]
    v_idx = [0*8+0, 0*8+1, 1*8+0, 1*8+1] # B[0,0], B[0,1], B[1,0], B[1,1]
    w_idx = [0*3+0, 1*3+0, 0*3+1, 1*3+1] # C[0,0], C[1,0], C[0,1], C[1,1] -> W[k*n+i]
    # W indices: k=0, i=0 -> 0; k=1, i=0 -> 3; k=0, i=1 -> 1; k=1, i=1 -> 4
    # My w_idx list order is [w00, w01, w10, w11] corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
    # So w00 -> W[0], w01 -> W[3], w10 -> W[1], w11 -> W[4]
    w_idx = [0, 3, 1, 4]
    add_strassen_block(u_idx, v_idx, w_idx, col)
    col += 7
    
    # Block 2: A_R * B_L10 (B rows 2,3 cols 0,1) -> C cols 0,1
    u_idx = [0*4+2, 0*4+3, 1*4+2, 1*4+3] # A[0,2], A[0,3], A[1,2], A[1,3]
    v_idx = [2*8+0, 2*8+1, 3*8+0, 3*8+1] # B[2,0], B[2,1], B[3,0], B[3,1]
    w_idx = [0, 3, 1, 4]
    add_strassen_block(u_idx, v_idx, w_idx, col)
    col += 7
    
    # Block 3: A_L * B_L01 (B rows 0,1 cols 2,3) -> C cols 2,3
    u_idx = [0, 1, 4, 5]
    v_idx = [0*8+2, 0*8+3, 1*8+2, 1*8+3] # B[0,2], B[0,3], B[1,2], B[1,3]
    # W indices for C cols 2,3:
    # C[0,2] -> W[2*3+0]=6, C[0,3] -> W[3*3+0]=9, C[1,2] -> W[2*3+1]=7, C[1,3] -> W[3*3+1]=10
    w_idx = [6, 9, 7, 10]
    add_strassen_block(u_idx, v_idx, w_idx, col)
    col += 7
    
    # Block 4: A_R * B_L11 (B rows 2,3 cols 2,3) -> C cols 2,3
    u_idx = [2, 3, 6, 7]
    v_idx = [2*8+2, 2*8+3, 3*8+2, 3*8+3] # B[2,2], B[2,3], B[3,2], B[3,3]
    w_idx = [6, 9, 7, 10]
    add_strassen_block(u_idx, v_idx, w_idx, col)
    col += 7
    
    # Block 5: A_L * B_R00 (B rows 0,1 cols 4,5) -> C cols 4,5
    u_idx = [0, 1, 4, 5]
    v_idx = [0*8+4, 0*8+5, 1*8+4, 1*8+5] # B[0,4], B[0,5], B[1,4], B[1,5]
    # W indices for C cols 4,5:
    # C[0,4] -> W[4*3+0]=12, C[0,5] -> W[5*3+0]=15, C[1,4] -> W[4*3+1]=13, C[1,5] -> W[5*3+1]=16
    w_idx = [12, 15, 13, 16]
    add_strassen_block(u_idx, v_idx, w_idx, col)
    col += 7
    
    # Block 6: A_R * B_R10 (B rows 2,3 cols 4,5) -> C cols 4,5
    u_idx = [2, 3, 6, 7]
    v_idx = [2*8+4, 2*8+5, 3*8+4, 3*8+5] # B[2,4], B[2,5], B[3,4], B[3,5]
    w_idx = [12, 15, 13, 16]
    add_strassen_block(u_idx, v_idx, w_idx, col)
    col += 7
    
    # Block 7: A_L * B_R01 (B rows 0,1 cols 6,7) -> C cols 6,7
    u_idx = [0, 1, 4, 5]
    v_idx = [0*8+6, 0*8+7, 1*8+6, 1*8+7] # B[0,6], B[0,7], B[1,6], B[1,7]
    # W indices for C cols 6,7:
    # C[0,6] -> W[6*3+0]=18, C[0,7] -> W[7*3+0]=21, C[1,6] -> W[6*3+1]=19, C[1,7] -> W[7*3+1]=22
    w_idx = [18, 21, 19, 22]
    add_strassen_block(u_idx, v_idx, w_idx, col)
    col += 7
    
    # Block 8: A_R * B_R11 (B rows 2,3 cols 6,7) -> C cols 6,7
    u_idx = [2, 3, 6, 7]
    v_idx = [2*8+6, 2*8+7, 3*8+6, 3*8+7] # B[2,6], B[2,7], B[3,6], B[3,7]
    w_idx = [18, 21, 19, 22]
    add_strassen_block(u_idx, v_idx, w_idx, col)
    col += 7
    
    # Bottom row of A (i=2)
    # Schoolbook for 1x4x8
    for j in range(4):
        for k in range(8):
            U[2*4 + j][col] = 1
            V[j*8 + k][col] = 1
            W[k*3 + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
