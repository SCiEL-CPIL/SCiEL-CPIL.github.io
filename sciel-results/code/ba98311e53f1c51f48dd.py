"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Strategy: 4+1 block split with rank-49 for 4x4*4x4 (Strassen at two levels)
    # Total rank: 49 + 45 = 94
    
    # Build the 4x4*4x4 rank-49 decomposition using Strassen at two levels.
    # 4x4 viewed as 2x2 blocks of 2x2 matrices.
    
    # Block-level Strassen terms: (x_blocks, y_blocks, z_blocks)
    block_strassen = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Within-block Strassen terms: (p_coeffs, q_coeffs, r_coeffs)
    within_strassen = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    U44 = [[0] * 49 for _ in range(16)]
    V44 = [[0] * 49 for _ in range(16)]
    W44 = [[0] * 49 for _ in range(16)]
    
    col = 0
    for m_idx in range(7):
        x_blocks, y_blocks, z_blocks = block_strassen[m_idx]
        for s_idx in range(7):
            p_coeffs, q_coeffs, r_coeffs = within_strassen[s_idx]
            
            for i in range(4):
                for j in range(4):
                    bi, bj = i // 2, j // 2
                    wi, wj = i % 2, j % 2
                    x_coeff = x_blocks.get((bi, bj), 0) * p_coeffs.get((wi, wj), 0)
                    if x_coeff != 0:
                        U44[i * 4 + j][col] = x_coeff
            
            for j in range(4):
                for k in range(4):
                    bi, bj = j // 2, k // 2
                    wi, wj = j % 2, k % 2
                    y_coeff = y_blocks.get((bi, bj), 0) * q_coeffs.get((wi, wj), 0)
                    if y_coeff != 0:
                        V44[j * 4 + k][col] = y_coeff
            
            for k in range(4):
                for i in range(4):
                    bi, bj = k // 2, i // 2
                    wi, wj = k % 2, i % 2
                    z_coeff = z_blocks.get((bi, bj), 0) * r_coeffs.get((wi, wj), 0)
                    if z_coeff != 0:
                        W44[k * 4 + i][col] = z_coeff
            
            col += 1
    
    total_rank = 94
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]
    
    for i in range(16):
        for j in range(49):
            U[i][j] = U44[i][j]
            V[i][j] = V44[i][j]
            W[i][j] = W44[i][j]
    
    col = 49
    
    # C12 = A11*B12 + A12*B22
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 5 + i][col] = 1
            col += 1
    
    for i in range(4):
        U[i * 5 + 4][col] = 1
        V[4 * 5 + 4][col] = 1
        W[4 * 5 + i][col] = 1
        col += 1
    
    # C21 = A21*B11 + A22*B21
    for j in range(4):
        for k in range(4):
            U[4 * 5 + j][col] = 1
            V[j * 5 + k][col] = 1
            W[k * 5 + 4][col] = 1
            col += 1
    
    for k in range(4):
        U[4 * 5 + 4][col] = 1
        V[4 * 5 + k][col] = 1
        W[k * 5 + 4][col] = 1
        col += 1
    
    # C22 = A21*B12 + A22*B22
    for j in range(4):
        U[4 * 5 + j][col] = 1
        V[j * 5 + 4][col] = 1
        W[4 * 5 + 4][col] = 1
        col += 1
    
    U[4 * 5 + 4][col] = 1
    V[4 * 5 + 4][col] = 1
    W[4 * 5 + 4][col] = 1
    col += 1
    
    assert col == total_rank
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
