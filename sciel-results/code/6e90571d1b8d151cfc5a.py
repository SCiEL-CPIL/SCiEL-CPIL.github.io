import time

DIMENSIONS = [2, 4, 5]

def construct():
    _t0 = time.time()
    _deadline = _t0 + 120.0
    
    n, m, p = 2, 4, 5
    
    # Rank 36 decomposition for M_{2,4,5}
    # Structure: Two independent M_{2,2,5} blocks.
    # Each M_{2,2,5} uses:
    #   - Strassen (rank 7) for k=0,1
    #   - Strassen (rank 7) for k=2,3
    #   - Schoolbook (rank 4) for k=4
    # Total rank = 2 * (7 + 7 + 4) = 36.
    
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    # Strassen coefficients for 2x2x2
    # Local indices: i,l in {0,1}; j,m in {0,1}; k,n in {0,1}
    # U[i*2+j], V[j*2+k], W[k*2+i]
    S_U = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    S_V = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    S_W = [
        [1, 0, 0, 1],  # M1
        [0, 1, 0, -1], # M2
        [0, 0, 1, 1],  # M3
        [1, 1, 0, 0],  # M4
        [-1, 0, 1, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ]
    
    # Schoolbook coefficients for 2x2x1
    # Local indices: i,l in {0,1}; j,m in {0,1}; k=0
    # U[i*2+j], V[j*2+0], W[0*2+i]
    SB_U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    SB_V = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ]
    SB_W = [
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base, col_start):
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    gi = i_base + li
                    gj = j_base + lj
                    U[gi * m + gj][col_start + r] = S_U[r][li * 2 + lj]
            for lj in range(2):
                for lk in range(2):
                    gj = j_base + lj
                    gk = k_base + lk
                    V[gj * p + gk][col_start + r] = S_V[r][lj * 2 + lk]
            for lk in range(2):
                for li in range(2):
                    gk = k_base + lk
                    gi = i_base + li
                    W[gk * n + gi][col_start + r] = S_W[r][lk * 2 + li]
        return col_start + 7

    def add_schoolbook_2x2x1_block(i_base, j_base, k_base, col_start):
        for r in range(4):
            for li in range(2):
                for lj in range(2):
                    gi = i_base + li
                    gj = j_base + lj
                    U[gi * m + gj][col_start + r] = SB_U[r][li * 2 + lj]
            for lj in range(2):
                gj = j_base + lj
                gk = k_base
                V[gj * p + gk][col_start + r] = SB_V[r][lj]
            for li in range(2):
                gk = k_base
                gi = i_base + li
                W[gk * n + gi][col_start + r] = SB_W[r][li]
        return col_start + 4

    # Block 1: j_base = 0 (uses A[:, 0:2] and B[0:2, :])
    col = add_strassen_block(0, 0, 0, col)       # k=0,1
    col = add_strassen_block(0, 0, 2, col)       # k=2,3
    col = add_schoolbook_2x2x1_block(0, 0, 4, col) # k=4
    
    # Block 2: j_base = 2 (uses A[:, 2:4] and B[2:4, :])
    col = add_strassen_block(0, 2, 0, col)       # k=0,1
    col = add_strassen_block(0, 2, 2, col)       # k=2,3
    col = add_schoolbook_2x2x1_block(0, 2, 4, col) # k=4
    
    assert col == rank, f"Column mismatch: {col} vs {rank}"
    
    return {"U": U, "V": V, "W": W}
