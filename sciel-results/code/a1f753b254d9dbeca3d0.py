"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    
    # Strassen 2x2 factors
    # U_r, V_r, W_r are 2x2 matrices
    # Indices: U[i,j], V[j,k], W[k,i]
    
    U_0 = [[1, 0], [0, 1]]
    V_0 = [[1, 0], [0, 1]]
    W_0 = [[1, 0], [0, 1]]
    
    U_1 = [[0, 0], [1, 1]]
    V_1 = [[1, 0], [0, 0]]
    W_1 = [[0, 0], [1, 1]]
    
    U_2 = [[1, 0], [0, 0]]
    V_2 = [[0, 1], [0, -1]]
    W_2 = [[0, 1], [0, 1]]
    
    U_3 = [[0, 0], [0, 1]]
    V_3 = [[-1, 0], [1, 0]]
    W_3 = [[0, 0], [1, 1]]
    
    U_4 = [[1, 1], [0, 0]]
    V_4 = [[0, 0], [0, 1]]
    W_4 = [[0, 0], [1, 1]]
    
    U_5 = [[-1, 0], [1, 0]]
    V_5 = [[1, 1], [0, 0]]
    W_5 = [[0, 0], [1, 1]]
    
    U_6 = [[0, 1], [0, -1]]
    V_6 = [[0, 0], [1, 1]]
    W_6 = [[0, 0], [1, 1]]
    
    U_blocks = [U_0, U_1, U_2, U_3, U_4, U_5, U_6]
    V_blocks = [V_0, V_1, V_2, V_3, V_4, V_5, V_6]
    W_blocks = [W_0, W_1, W_2, W_3, W_4, W_5, W_6]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: k in 0,1
    for r in range(7):
        u_mat = U_blocks[r]
        v_mat = V_blocks[r]
        w_mat = W_blocks[r]
        
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = u_mat[i][j]
        
        for j in range(m):
            for k in range(2): # k=0,1
                V[j * p + k][r] = v_mat[j][k]
        
        for k in range(2): # k=0,1
            for i in range(n):
                W[k * n + i][r] = w_mat[k][i]
                
    # Block 2: k in 2,3
    for r in range(7):
        u_mat = U_blocks[r]
        v_mat = V_blocks[r]
        w_mat = W_blocks[r]
        
        col = r + 7
        
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = u_mat[i][j]
        
        for j in range(m):
            for k in range(2): # k=0,1 local
                V[j * p + (k + 2)][col] = v_mat[j][k]
        
        for k in range(2): # k=0,1 local
            for i in range(n):
                W[(k + 2) * n + i][col] = w_mat[k][i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
