import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Block size 3, 2x2 blocks
    block_size = 3
    n_blocks = 2
    
    # Rank is 7 (Strassen steps) * 27 (schoolbook 3x3) = 189
    rank = 189
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Strassen step definitions
    # Indices: 0=A11, 1=A12, 2=A21, 3=A22
    U_steps = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    V_steps = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    W_steps = [
        [1, 0, 0, 1], # M1 -> C11, C22
        [0, 0, 1, -1], # M2 -> C21, -C22
        [0, 1, 0, 1], # M3 -> C12, C22
        [1, 0, 1, 0], # M4 -> C11, C21
        [-1, 1, 0, 0], # M5 -> -C11, C12
        [0, 0, 0, 1], # M6 -> C22
        [1, 0, 0, 0]  # M7 -> C11
    ]
    
    # Helper to map block index and internal indices to global index
    def global_idx(block, i, j, dim):
        # block: 0-3 (0:00, 1:01, 2:10, 3:11)
        br = block // 2
        bc = block % 2
        global_row = br * block_size + i
        global_col = bc * block_size + j
        return global_row * dim + global_col

    col = 0
    for r in range(7):
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    # U part:
                    # U'_s is 1 at (i, k) in 3x3
                    # So we need to add U_steps[r][b] to U[global_idx(b, i, k, m)] for all b
                    for b in range(4):
                        coeff = U_steps[r][b]
                        if coeff != 0:
                            idx = global_idx(b, i, k, m)
                            U[idx, col] = coeff

                    # V part:
                    # V'_s is 1 at (k, j) in 3x3
                    for b in range(4):
                        coeff = V_steps[r][b]
                        if coeff != 0:
                            idx = global_idx(b, k, j, p)
                            V[idx, col] = coeff

                    # W part:
                    # W'_s is 1 at (i, j) in 3x3
                    for b in range(4):
                        coeff = W_steps[r][b]
                        if coeff != 0:
                            idx = global_idx(b, i, j, n)
                            W[idx, col] = coeff

                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
