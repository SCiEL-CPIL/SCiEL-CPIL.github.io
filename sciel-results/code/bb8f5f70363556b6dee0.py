"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    # Partition p=14 columns into 7 blocks of 2 columns each
    num_blocks = p // 2  # 7 blocks
    rank = 7 * num_blocks  # 7 Strassen mults per block * 7 blocks = 49
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 49)
    V = [[0] * rank for _ in range(m * p)]  # (28, 49)
    W = [[0] * rank for _ in range(p * n)]  # (28, 49)
    
    for block in range(num_blocks):
        col_offset = 7 * block
        # B column indices for this block
        b_col0 = 2 * block      # first column of block
        b_col1 = 2 * block + 1  # second column of block
        
        # V indices for B elements in this block
        # B[i, k] is at V[i*p + k, col]
        v_e = 0 * p + b_col0   # B[0, b_col0]
        v_f = 0 * p + b_col1   # B[0, b_col1]
        v_g = 1 * p + b_col0   # B[1, b_col0]
        v_h = 1 * p + b_col1   # B[1, b_col1]
        
        # W indices for output C[i, k] at W[k*n + i, col]
        # Output columns for this block: b_col0 and b_col1
        w_c00 = b_col0 * n + 0  # C[0, b_col0]
        w_c01 = b_col1 * n + 0  # C[0, b_col1]
        w_c10 = b_col0 * n + 1  # C[1, b_col0]
        w_c11 = b_col1 * n + 1  # C[1, b_col1]
        
        # U indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        u_a = 0  # A[0,0]
        u_b = 1  # A[0,1]
        u_c = 2  # A[1,0]
        u_d = 3  # A[1,1]
        
        # Strassen factors for this block
        # M1 = (a+d)(e+h)
        col = col_offset + 0
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        W[w_c00][col] += 1
        W[w_c11][col] += 1
        
        # M2 = (c+d)e
        col = col_offset + 1
        U[u_c][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        W[w_c10][col] += 1
        
        # M3 = a(f-h)
        col = col_offset + 2
        U[u_a][col] += 1
        V[v_f][col] += 1
        V[v_h][col] += -1
        W[w_c01][col] += 1
        W[w_c11][col] += -1
        
        # M4 = d(g-e)
        col = col_offset + 3
        U[u_d][col] += 1
        V[v_g][col] += 1
        V[v_e][col] += -1
        W[w_c00][col] += -1
        W[w_c10][col] += 1
        
        # M5 = (a+b)h
        col = col_offset + 4
        U[u_a][col] += 1
        U[u_b][col] += 1
        V[v_h][col] += 1
        W[w_c00][col] += 1
        W[w_c01][col] += 1
        
        # M6 = (c-a)(e+f)
        col = col_offset + 5
        U[u_c][col] += 1
        U[u_a][col] += -1
        V[v_e][col] += 1
        V[v_f][col] += 1
        W[w_c11][col] += 1
        
        # M7 = (b-d)(g+h)
        col = col_offset + 6
        U[u_b][col] += 1
        U[u_d][col] += -1
        V[v_g][col] += 1
        V[v_h][col] += 1
        W[w_c00][col] += 1
        W[w_c11][col] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
