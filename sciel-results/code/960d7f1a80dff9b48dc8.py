import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    
    # Helper to build Strassen decomposition for n x n x n
    def strassen_decomp(size):
        if size == 1:
            return [[1]], [[1]], [[1]]
        
        half = size // 2
        U_sub, V_sub, W_sub = strassen_decomp(half)
        rank_sub = len(U_sub[0])
        
        U_new = [[0] * (7 * rank_sub) for _ in range(size * size)]
        V_new = [[0] * (7 * rank_sub) for _ in range(size * size)]
        W_new = [[0] * (7 * rank_sub) for _ in range(size * size)]
        
        # Specs for the 7 Strassen products
        # Each spec: (list of (block_row, block_col, sign) for A, ... for B, ... for C)
        # Blocks are 0,0 (11), 0,1 (12), 1,0 (21), 1,1 (22)
        
        specs = [
            # M1 = (A11 + A22)(B11 + B22)
            {
                'A': [((0,0), 1), ((1,1), 1)],
                'B': [((0,0), 1), ((1,1), 1)],
                'C': [((0,0), 1), ((1,1), -1)] # C11 gets M1, C22 gets -M1? No.
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M2 + M3 + M6
            },
            # M2 = (A21 + A22)B11
            {
                'A': [((1,0), 1), ((1,1), 1)],
                'B': [((0,0), 1)],
                'C': [((1,0), 1), ((1,1), -1)]
            },
            # M3 = A11(B12 - B22)
            {
                'A': [((0,0), 1)],
                'B': [((0,1), 1), ((1,1), -1)],
                'C': [((0,1), 1), ((1,1), 1)]
            },
            # M4 = A22(B21 - B11)
            {
                'A': [((1,1), 1)],
                'B': [((1,0), 1), ((0,0), -1)],
                'C': [((0,0), 1), ((1,0), 1)]
            },
            # M5 = (A11 + A12)B22
            {
                'A': [((0,0), 1), ((0,1), 1)],
                'B': [((1,1), 1)],
                'C': [((0,0), -1), ((0,1), 1)]
            },
            # M6 = (A21 - A11)(B11 + B12)
            {
                'A': [((1,0), 1), ((0,0), -1)],
                'B': [((0,0), 1), ((0,1), 1)],
                'C': [((1,1), 1)]
            },
            # M7 = (A12 - A22)(B21 + B22)
            {
                'A': [((0,1), 1), ((1,1), -1)],
                'B': [((1,0), 1), ((1,1), 1)],
                'C': [((0,0), 1)]
            },
        ]
        
        for t in range(7):
            col_offset = t * rank_sub
            spec = specs[t]
            
            for r in range(rank_sub):
                # Iterate through non-zero entries in sub-decomp? 
                # U_sub is list of lists. U_sub[row][r]
                # We need to add U_sub[row][r] to U_new[global_row][col_offset + r]
                
                # Get non-zero rows in U_sub for column r
                # Actually, U_sub is (half*half) x rank_sub.
                # So U_sub[i][r] is the value.
                
                for i in range(half * half):
                    val_u = U_sub[i][r]
                    if val_u != 0:
                        for (br, bc), sign in spec['A']:
                            # Global row index for A
                            # A is size x size.
                            # Block br, bc of size half.
                            # Local row i corresponds to local (lr, lc) where i = lr*half + lc.
                            lr, lc = divmod(i, half)
                            gr = br * half + lr
                            gc = bc * half + lc
                            global_row = gr * size + gc
                            U_new[global_row][col_offset + r] += val_u * sign
                            
                    val_v = V_sub[i][r]
                    if val_v != 0:
                        for (br, bc), sign in spec['B']:
                            lr, lc = divmod(i, half)
                            gr = br * half + lr
                            gc = bc * half + lc
                            global_row = gr * size + gc
                            V_new[global_row][col_offset + r] += val_v * sign
                            
                    val_w = W_sub[i][r]
                    if val_w != 0:
                        for (br, bc), sign in spec['C']:
                            lr, lc = divmod(i, half)
                            gr = br * half + lr
                            gc = bc * half + lc
                            global_row = gr * size + gc
                            W_new[global_row][col_offset + r] += val_w * sign

        return U_new, V_new, W_new

    # Get Strassen for 4x4x4
    U_4, V_4, W_4 = strassen_decomp(4)
    rank_4 = len(U_4[0]) # 49
    
    # Initialize full matrices
    # U: 4x5 = 20 rows
    # V: 5x5 = 25 rows
    # W: 5x4 = 20 rows
    total_rank = rank_4 + 20 + 16
    U = [[0] * total_rank for _ in range(20)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(20)]
    
    # Map Strassen 4x4x4 to 4x5x5
    # Strassen indices: i,j,k in 0..3
    # U_4 rows: i*4 + j
    # V_4 rows: j*4 + k
    # W_4 rows: k*4 + i
    
    for i in range(4):
        for j in range(4):
            u_idx_4 = i * 4 + j
            u_idx_full = i * 5 + j
            for r in range(rank_4):
                U[u_idx_full][r] = U_4[u_idx_4][r]
                
            for k in range(4):
                v_idx_4 = j * 4 + k
                v_idx_full = j * 5 + k
                V[v_idx_full][r] = V_4[v_idx_4][r]
                
            for k in range(4):
                w_idx_4 = k * 4 + i
                w_idx_full = k * 4 + i
                W[w_idx_full][r] = W_4[w_idx_4][r]

    # Add Term 2: A_i4 * B_4k (i in 0..3, k in 0..4)
    # Rank 20
    col_offset = rank_4
    for i in range(4):
        for k in range(5):
            u_idx = i * 5 + 4
            v_idx = 4 * 5 + k
            w_idx = k * 4 + i
            U[u_idx][col_offset] = 1
            V[v_idx][col_offset] = 1
            W[w_idx][col_offset] = 1
            col_offset += 1
            
    # Add Term 3: A_ij * B_j4 (i in 0..3, j in 0..3)
    # Rank 16
    for i in range(4):
        for j in range(4):
            u_idx = i * 5 + j
            v_idx = j * 5 + 4
            w_idx = 4 * 4 + i
            U[u_idx][col_offset] = 1
            V[v_idx][col_offset] = 1
            W[w_idx][col_offset] = 1
            col_offset += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
