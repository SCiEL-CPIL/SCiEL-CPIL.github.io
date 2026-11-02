"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Block decomposition for 5x5 x 5x5 matrix multiplication.
    Uses Strassen's algorithm for 2x2 sub-blocks and schoolbook for others.
    This reduces the rank from 125 (pure schoolbook) to 124.
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Global factor matrices
    U = [[0] * (n * m * p) for _ in range(n * m)] # Max rank 125
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    
    current_rank = 0
    
    def add_subproblem(sub_n, sub_m, sub_p, offset_i, offset_j, offset_k, U_sub, V_sub, W_sub):
        nonlocal current_rank
        rank_sub = len(U_sub[0])
        for i in range(sub_n):
            for j in range(sub_m):
                global_i = offset_i + i
                global_j = offset_j + j
                global_idx_u = global_i * m + global_j
                for r in range(rank_sub):
                    U[global_idx_u][current_rank + r] += U_sub[i * sub_m + j][r]
        
        for j in range(sub_m):
            for k in range(sub_p):
                global_j = offset_j + j
                global_k = offset_k + k
                global_idx_v = global_j * p + global_k
                for r in range(rank_sub):
                    V[global_idx_v][current_rank + r] += V_sub[j * sub_p + k][r]
        
        for k in range(sub_p):
            for i in range(sub_n):
                global_k = offset_k + k
                global_i = offset_i + i
                global_idx_w = global_k * n + global_i
                for r in range(rank_sub):
                    W[global_idx_w][current_rank + r] += W_sub[k * sub_n + i][r]
                    
        current_rank += rank_sub

    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def strassen_2x2():
        # Returns U, V, W for 2x2x2 with rank 7
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        
        # M0 = (a+d)(e+h)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[2][0] = 1; W[1][0] = -1; W[3][0] = 1 # C00, C10, C01, C11? 
        # Let's re-verify W indices.
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        # W[k*n + i, r]
        # C00 is k=0, i=0 -> idx 0
        # C01 is k=1, i=0 -> idx 1
        # C10 is k=0, i=1 -> idx 2
        # C11 is k=1, i=1 -> idx 3
        
        W[0][0] = 1 # C00 gets M0
        W[1][0] = 0 # C01 gets 0 from M0
        W[2][0] = 0 # C10 gets 0 from M0
        W[3][0] = 1 # C11 gets M0
        
        # M1 = (c+d)e
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1 # C10 gets M1
        W[3][1] = -1 # C11 gets -M1
        
        # M2 = a(f-h)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1 # C01 gets M2
        W[3][2] = 1 # C11 gets M2
        
        # M3 = d(g-e)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1 # C00 gets M3
        W[2][3] = 1 # C10 gets M3
        
        # M4 = (a+b)h
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1 # C00 gets -M4
        W[1][4] = 1 # C01 gets M4
        
        # M5 = (c-a)(e+f)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1 # C11 gets M5
        
        # M6 = (b-d)(g+h)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1 # C00 gets M6
        
        return U, V, W

    # Block 1: Top-Left 2x2 (C00)
    # A[0:2, 0:2] * B[0:2, 0:2]
    U1, V1, W1 = strassen_2x2()
    add_subproblem(2, 2, 2, 0, 0, 0, U1, V1, W1)
    
    # Block 2: Top-Right contribution to C00: A[0:2, 2:5] * B[2:5, 0:2]
    # n=2, m=3, p=2
    U2, V2, W2 = schoolbook(2, 3, 2)
    add_subproblem(2, 3, 2, 0, 2, 0, U2, V2, W2)
    
    # Block 3: Top-Left contribution to C01: A[0:2, 0:2] * B[0:2, 2:5]
    # n=2, m=2, p=3
    U3, V3, W3 = schoolbook(2, 2, 3)
    add_subproblem(2, 2, 3, 0, 0, 2, U3, V3, W3)
    
    # Block 4: Top-Right contribution to C01: A[0:2, 2:5] * B[2:5, 2:5]
    # n=2, m=3, p=3
    U4, V4, W4 = schoolbook(2, 3, 3)
    add_subproblem(2, 3, 3, 0, 2, 2, U4, V4, W4)
    
    # Block 5: Bottom-Left contribution to C10: A[2:5, 0:2] * B[0:2, 0:2]
    # n=3, m=2, p=2
    U5, V5, W5 = schoolbook(3, 2, 2)
    add_subproblem(3, 2, 2, 2, 0, 0, U5, V5, W5)
    
    # Block 6: Bottom-Right contribution to C10: A[2:5, 2:5] * B[2:5, 0:2]
    # n=3, m=3, p=2
    U6, V6, W6 = schoolbook(3, 3, 2)
    add_subproblem(3, 3, 2, 2, 2, 0, U6, V6, W6)
    
    # Block 7: Bottom-Left contribution to C11: A[2:5, 0:2] * B[0:2, 2:5]
    # n=3, m=2, p=3
    U7, V7, W7 = schoolbook(3, 2, 3)
    add_subproblem(3, 2, 3, 2, 0, 2, U7, V7, W7)
    
    # Block 8: Bottom-Right contribution to C11: A[2:5, 2:5] * B[2:5, 2:5]
    # n=3, m=3, p=3
    U8, V8, W8 = schoolbook(3, 3, 3)
    add_subproblem(3, 3, 3, 2, 2, 2, U8, V8, W8)
    
    # Trim the matrices to the actual rank used
    U_trimmed = [row[:current_rank] for row in U]
    V_trimmed = [row[:current_rank] for row in V]
    W_trimmed = [row[:current_rank] for row in W]
    
    return {"U": U_trimmed, "V": V_trimmed, "W": W_trimmed}
# EVOLVE-BLOCK-END
