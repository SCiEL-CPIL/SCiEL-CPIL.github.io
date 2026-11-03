"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition:
    # Split p=10 into 5 blocks of size 2.
    # For each block, use Strassen (rank 7) for the 2x2x2 part (rows 0,1 of B)
    # and standard multiplication (rank 4) for the 2x1x2 part (row 2 of B).
    # Total rank = 5 * (7 + 4) = 55.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # Indices: 0:00, 1:01, 2:10, 3:11 (local)
    # Global U rows for A1 (cols 0,1):
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 3
    # i=1, j=1 -> 4
    
    # Strassen linear forms for B (2x2)
    # Indices: 0:00, 1:01, 2:10, 3:11 (local)
    
    # Strassen output matrices W_t (2x2)
    # t=0: M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # t=1: M2 = (A10+A11)B00 -> C10+=1, C11-=1
    # t=2: M3 = A00(B10-B11) -> C01+=1, C11+=1
    # t=3: M4 = A11(B01-B00) -> C00+=1
    # t=4: M5 = (A00+A10)B11 -> C00-=1, C10+=1
    # t=5: M6 = (A11-A00)(B00+B10) -> C11+=1
    # t=6: M7 = (A01-A11)(B11+B01) -> C00-=1, C01+=1
    
    strassen_U_forms = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 0, 1, 0], # M5
        [-1, 0, 0, 1],# M6
        [0, 1, 0, -1] # M7
    ]
    
    strassen_V_forms = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [0, 1, 0, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 0, 1, 0], # M6
        [0, 1, 0, 1]  # M7
    ]
    
    # W matrices for Strassen terms (entries at local indices)
    # local (i, k) -> 0:00, 1:01, 2:10, 3:11
    strassen_W_mats = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, -1]],# M2
        [[0, 1], [0, 1]], # M3
        [[1, 0], [0, 0]], # M4
        [[-1, 0], [1, 0]],# M5
        [[0, 0], [0, 1]], # M6
        [[-1, 1], [0, 0]] # M7
    ]
    
    # Standard multiplication for 2x1 x 1x2
    # Terms:
    # s=0: A02 * B20 -> C00 += 1
    # s=1: A02 * B21 -> C01 += 1
    # s=2: A12 * B20 -> C10 += 1
    # s=3: A12 * B21 -> C11 += 1
    
    std_U_rows = [2, 2, 5, 5] # A02, A02, A12, A12
    std_V_cols = [0, 1, 0, 1] # Local col index in B block (0 or 1)
    std_W_mats = [
        [[1, 0], [0, 0]], # s=0
        [[0, 1], [0, 0]], # s=1
        [[0, 0], [1, 0]], # s=2
        [[0, 0], [0, 1]]  # s=3
    ]
    
    for b in range(5):
        base_col = b * 11
        local_cols = [2 * b, 2 * b + 1]
        
        # Strassen part
        for t in range(7):
            r = base_col + t
            
            # U assignments
            # local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # global U row: i*m + j
            u_vals = strassen_U_forms[t]
            # idx 0: i=0,j=0 -> row 0
            U[0][r] += u_vals[0]
            # idx 1: i=0,j=1 -> row 1
            U[1][r] += u_vals[1]
            # idx 2: i=1,j=0 -> row 3
            U[3][r] += u_vals[2]
            # idx 3: i=1,j=1 -> row 4
            U[4][r] += u_vals[3]
            
            # V assignments
            # local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # global V row: j*p + k
            v_vals = strassen_V_forms[t]
            # idx 0: j=0, k=2b -> row 0*10 + 2b
            V[0 * p + local_cols[0]][r] += v_vals[0]
            # idx 1: j=0, k=2b+1 -> row 0*10 + 2b+1
            V[0 * p + local_cols[1]][r] += v_vals[1]
            # idx 2: j=1, k=2b -> row 1*10 + 2b
            V[1 * p + local_cols[0]][r] += v_vals[2]
            # idx 3: j=1, k=2b+1 -> row 1*10 + 2b+1
            V[1 * p + local_cols[1]][r] += v_vals[3]
            
            # W assignments
            # local W matrix 2x2
            w_mat = strassen_W_mats[t]
            # local (i, k_local)
            # global W row: k*n + i
            # k = local_cols[k_local]
            for i in range(2):
                for k_local in range(2):
                    val = w_mat[i][k_local]
                    if val != 0:
                        k_global = local_cols[k_local]
                        w_row = k_global * n + i
                        W[w_row][r] += val

        # Standard part
        for s in range(4):
            r = base_col + 7 + s
            
            # U assignment
            u_row = std_U_rows[s]
            U[u_row][r] = 1
            
            # V assignment
            v_row = 2 * p + local_cols[std_V_cols[s]]
            V[v_row][r] = 1
            
            # W assignment
            w_mat = std_W_mats[s]
            for i in range(2):
                for k_local in range(2):
                    val = w_mat[i][k_local]
                    if val != 0:
                        k_global = local_cols[k_local]
                        w_row = k_global * n + i
                        W[w_row][r] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
