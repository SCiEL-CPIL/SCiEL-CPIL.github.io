import numpy as np

def construct():
    n, m, p = 2, 4, 5
    
    # Rank 32 decomposition
    # Split k into {0,1,2,3} and {4}
    # 2x4x4: four 2x2x2 Strassen blocks (j=0,1 and j=2,3; k=0,1 and k=2,3)
    # 2x4x1: schoolbook rank 4
    # Total: 4*7 + 4 = 32
    
    rank = 32
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2x2
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Schoolbook coefficients for 2x4x1 (multiplying 2x4 by 4x1)
    # U: (2*4, 4) = (8, 4) - identity-like
    # V: (4*1, 4) = (4, 4) - identity
    # W: (1*2, 4) = (2, 4) - for each output (i), sum over j
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base, col_start):
        nonlocal col
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    gi = i_base + li
                    gj = j_base + lj
                    U[gi * m + gj, col_start + r] = S_U[r][li * 2 + lj]
            for lj in range(2):
                for lk in range(2):
                    gj = j_base + lj
                    gk = k_base + lk
                    V[gj * p + gk, col_start + r] = S_V[r][lj * 2 + lk]
            for lk in range(2):
                for li in range(2):
                    gk = k_base + lk
                    gi = i_base + li
                    W[gk * n + gi, col_start + r] = S_W[r][lk * 2 + li]
        col = col_start + 7
        return col
    
    def add_schoolbook_2x4x1_block(i_base, j_base, k_base, col_start):
        nonlocal col
        # 2x4x1: C[i,k] = sum_j A[i,j] * B[j,k]
        # For fixed k_base, we need rank 4 terms
        for r in range(4):
            # U: A[i, j_base + r]
            for li in range(2):
                gi = i_base + li
                gj = j_base + r
                U[gi * m + gj, col_start + r] = 1
            # V: B[j_base + r, k_base]
            gj = j_base + r
            gk = k_base
            V[gj * p + gk, col_start + r] = 1
            # W: C[k_base, i] for all i
            for li in range(2):
                gk = k_base
                gi = i_base + li
                W[gk * n + gi, col_start + r] = 1
        col = col_start + 4
        return col
    
    # Four 2x2x2 Strassen blocks for k in {0,1,2,3}
    add_strassen_block(0, 0, 0, col)  # j=0,1; k=0,1
    add_strassen_block(0, 2, 0, col)  # j=2,3; k=0,1
    add_strassen_block(0, 0, 2, col)  # j=0,1; k=2,3
    add_strassen_block(0, 2, 2, col)  # j=2,3; k=2,3
    
    # One 2x4x1 schoolbook block for k=4
    add_schoolbook_2x4x1_block(0, 0, 4, col)
    
    assert col == rank, f"Column mismatch: {col} vs {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
