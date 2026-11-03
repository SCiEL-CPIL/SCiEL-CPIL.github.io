import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define 2x2 Strassen factors
    # U, V, W are 2x2 matrices for r=0..6
    U2 = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, 0]],  # M2
        [[0, 1], [0, -1]], # M3
        [[1, 0], [1, 0]],  # M4
        [[-1, 1], [0, 0]], # M5
        [[0, 0], [0, 1]],  # M6
        [[1, 0], [0, 1]]   # M7
    ]
    V2 = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, 1]],  # M2
        [[1, 0], [0, 0]],  # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [-1, 0]]  # M7
    ]
    W2 = [
        [[1, 0], [0, 1]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 1], [-1, 0]], # M3
        [[-1, 0], [1, 0]], # M4
        [[0, 0], [0, 1]],  # M5
        [[1, 1], [0, 0]],  # M6
        [[0, 0], [1, 1]]   # M7
    ]

    # Generate 49 factors for 4x4x4
    U4 = []
    V4 = []
    W4 = []
    
    for r in range(7):
        for s in range(7):
            u_rs = np.kron(np.array(U2[r]), np.array(U2[s]))
            v_rs = np.kron(np.array(V2[r]), np.array(V2[s]))
            w_rs = np.kron(np.array(W2[r]), np.array(W2[s]))
            U4.append(u_rs.flatten())
            V4.append(v_rs.flatten())
            W4.append(w_rs.flatten())

    # Construct U, V, W for 4x4x8
    # Rank 98
    rank = 98
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # First 49 terms: columns 0-3 of B, rows 0-3 of C
    for c in range(49):
        # U
        for idx in range(16):
            U[idx][c] = U4[c][idx]
        
        # V: rows j*8 + k, k in 0..3
        for j in range(4):
            for k in range(4):
                V[j * 8 + k][c] = V4[c][j * 4 + k]
        
        # W: rows k*4 + i, k in 0..3
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][c] = W4[c][k * 4 + i]

    # Second 49 terms: columns 4-7 of B, rows 4-7 of C
    for c in range(49):
        col = c + 49
        # U
        for idx in range(16):
            U[idx][col] = U4[c][idx]
        
        # V: rows j*8 + k, k in 4..7
        for j in range(4):
            for k in range(4):
                V[j * 8 + (k + 4)][col] = V4[c][j * 4 + k]
        
        # W: rows k*4 + i, k in 4..7
        for k in range(4):
            for i in range(4):
                W[(k + 4) * 4 + i][col] = W4[c][k * 4 + i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
