import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5

    # Build a rank-85 decomposition of the 4x5 x 5x5 -> 4x5 multiplication tensor.
    # Strategy: split the 5x5 middle/output dimension into 4+1 and use:
    #   Part 1: 4x4x4 block (rank 49 via Kronecker of two 2x2 Strassen tensors)
    #   Part 2: 4x4x1 block (rank 16, naive)
    #   Part 3: 4x1x4 block (rank 16, naive)
    #   Part 4: 4x1x1 block (rank 4, naive)
    # Total rank: 49 + 16 + 16 + 4 = 85

    rank = 85
    U = [[0] * rank for _ in range(n * m)]   # 20 x 85
    V = [[0] * rank for _ in range(m * p)]   # 25 x 85
    W = [[0] * rank for _ in range(p * n)]   # 20 x 85

    # --- Part 1: 4x4x4 using Kronecker product of two 2x2 Strassen tensors (rank 49) ---
    # Strassen 2x2x2 factors (4x7 each)
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0, -1]
    ]
    S_V = [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, -1]
    ]
    S_W = [
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1]
    ]

    col = 0
    for a in range(7):
        for b in range(7):
            c = a * 7 + b
            for i1 in range(2):
                for i2 in range(2):
                    i = 2 * i1 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = 2 * j1 + j2
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = 2 * k1 + k2
                                    val = S_U[i1 * 2 + i2][a] * S_U[i1 * 2 + i2][b]
                                    # Actually, the Kronecker product structure:
                                    # U_4x4[i, c] = S_U[i1, a] * S_U[i2, b] where i = 2*i1 + i2
                                    # Let me redo this properly.
                                    pass

    # Let me redo the Kronecker product construction properly.
    # For 4x4x4, index i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # The term (a,b) in the Kronecker product corresponds to:
    # U[i, c] = S_U[i1, a] * S_U[i2, b]
    # V[j, c] = S_V[j1, a] * S_V[j2, b]
    # W[k, c] = S_W[k1, a] * S_W[k2, b]
    # where c = a*7 + b

    col = 0
    for a in range(7):
        for b in range(7):
            c = a * 7 + b
            for i1 in range(2):
                for i2 in range(2):
                    i = 2 * i1 + i2
                    u_val = S_U[i1][a] * S_U[i2][b]
                    if u_val != 0:
                        U[i * m + 0][col] += u_val  # j starts at 0 for the 4x4 block
            for j1 in range(2):
                for j2 in range(2):
                    j = 2 * j1 + j2
                    v_val = S_V[j1][a] * S_V[j2][b]
                    if v_val != 0:
                        V[j * p + 0][col] += v_val  # k starts at 0 for the 4x4 block
            for k1 in range(2):
                for k2 in range(2):
                    k = 2 * k1 + k2
                    w_val = S_W[k1][a] * S_W[k2][b]
                    if w_val != 0:
                        W[k * n + 0][col] += w_val  # i starts at 0 for the 4x4 block
            col += 1
    # col should be 49

    # --- Part 2: 4x4x1 block (rank 16, naive) ---
    # C[i, 4] = sum_{j=0}^{3} A[i, j] * B[j, 4]
    # U[i*5 + j, col] = 1 for i in 0..3, j in 0..3
    # V[j*5 + 4, col] = 1 for j in 0..3
    # W[4*5 + i, col] = 1 for i in 0..3
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    # col should be 49 + 16 = 65

    # --- Part 3: 4x1x4 block (rank 16, naive) ---
    # C[i, k] = A[i, 4] * B[4, k] for i in 0..3, k in 0..3
    # U[i*5 + 4, col] = 1
    # V[4*5 + k, col] = 1
    # W[k*5 + i, col] = 1
    for i in range(4):
        for k in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col should be 65 + 16 = 81

    # --- Part 4: 4x1x1 block (rank 4, naive) ---
    # C[i, 4] = A[i, 4] * B[4, 4] for i in 0..3
    # U[i*5 + 4, col] = 1
    # V[4*5 + 4, col] = 1
    # W[4*5 + i, col] = 1
    for i in range(4):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1
    # col should be 81 + 4 = 85

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
