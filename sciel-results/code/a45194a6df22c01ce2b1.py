"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition approach:
    # M_{2,4,7} = M_{2,4,4} + M_{2,4,3}
    # M_{2,4,4} = 2 * M_{2,2,4} = 2 * (2 * M_{2,2,2}) = 4 * M_{2,2,2}
    # M_{2,4,3} = 2 * M_{2,2,3} = 2 * (M_{2,2,2} + M_{2,2,1})
    
    # For M_{2,2,2}, we use the rank-7 Winograd decomposition.
    # For M_{2,2,1}, we use the rank-4 schoolbook decomposition.
    
    # Total rank = 4 * 7 + 2 * (7 + 4) = 28 + 22 = 50
    
    # Winograd 2x2x2 decomposition (rank 7):
    # The 7 products are:
    # p1 = (a11 + a22)(b11 + b22)
    # p2 = a12 b21
    # p3 = a21 b12
    # p4 = (a11 + a12) b11
    # p5 = (a22 + a21) b22
    # p6 = (a11 - a21) b21
    # p7 = (a22 - a12) b12
    
    # U matrix (4x7) for A:
    # Row 0 (a11): [1, 0, 0, 1, 0, 1, 0]
    # Row 1 (a12): [0, 1, 0, 1, 0, 0, 1]
    # Row 2 (a21): [0, 0, 1, 0, 1, 1, 0]
    # Row 3 (a22): [1, 0, 0, 0, 1, 0, 1]
    
    # V matrix (4x7) for B:
    # Row 0 (b11): [1, 0, 0, 1, 0, 0, 0]
    # Row 1 (b12): [0, 0, 1, 0, 0, 0, 1]
    # Row 2 (b21): [0, 1, 0, 0, 0, 1, 0]
    # Row 3 (b22): [1, 0, 0, 0, 1, 0, 0]
    
    # W matrix (4x7) for C:
    # C11 = p1 - p4 + p6
    # C12 = p2 + p4 - p7
    # C21 = p3 + p5 - p6
    # C22 = p1 - p3 + p5
    
    U_wino = [
        [1, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 0, 1, 0, 1]
    ]
    
    V_wino = [
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0]
    ]
    
    W_wino = [
        [1, 0, 0, -1, 0, 1, 0],   # C11 = p1 - p4 + p6
        [0, 1, 0, 1, 0, 0, -1],   # C12 = p2 + p4 - p7
        [0, 0, 1, 0, 1, -1, 0],   # C21 = p3 + p5 - p6
        [1, 0, -1, 0, 1, 0, 0]    # C22 = p1 - p3 + p5
    ]
    
    # Schoolbook for M_{2,2,1} (rank 4):
    # U (4x4), V (2x4), W (2x4)
    U_wino_2x2x1 = [
        [1, 0, 0, 0],  # a11
        [0, 1, 0, 0],  # a12
        [0, 0, 1, 0],  # a21
        [0, 0, 0, 1]   # a22
    ]
    V_wino_2x2x1 = [
        [1, 0, 0, 0],  # b11
        [0, 0, 1, 0]   # b21
    ]
    W_wino_2x2x1 = [
        [1, 0, 0, 0],  # c11
        [0, 0, 1, 0]   # c21
    ]
    
    # Now we build the full decomposition for M_{2,4,7}
    # We have 4 blocks of M_{2,2,2} and 2 blocks of M_{2,2,1}
    # Total rank = 4 * 7 + 2 * 4 = 28 + 8 = 36
    
    # Wait, let me recalculate:
    # M_{2,4,4} = 4 * M_{2,2,2} -> rank 4 * 7 = 28
    # M_{2,4,3} = 2 * M_{2,2,3} = 2 * (M_{2,2,2} + M_{2,2,1}) -> rank 2 * (7 + 4) = 22
    # Total = 28 + 22 = 50
    
    # Let's build the matrices
    total_rank = 50
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 50
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x 50
    W = [[0] * total_rank for _ in range(p * n)]  # 14 x 50
    
    col_offset = 0
    
    # Block 1: M_{2,2,2} for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    for i in range(4):
        for j in range(7):
            U[i * 2 + 0][col_offset + j] = U_wino[i][j]
            V[(0 * 2 + 0) * 7 + i][col_offset + j] = V_wino[i][j]
            W[(0 * 2 + 0) * 2 + i][col_offset + j] = W_wino[i][j]
    col_offset += 7
    
    # Block 2: M_{2,2,2} for A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    for i in range(4):
        for j in range(7):
            U[i * 2 + 2][col_offset + j] = U_wino[i][j]
            V[(2 * 2 + 0) * 7 + i][col_offset + j] = V_wino[i][j]
            W[(0 * 2 + 0) * 2 + i][col_offset + j] += W_wino[i][j]
    col_offset += 7
    
    # Block 3: M_{2,2,2} for A[0:2,0:2] * B[0:2,2:4] -> C[2:4,0:2]
    for i in range(4):
        for j in range(7):
            U[i * 2 + 0][col_offset + j] += U_wino[i][j]
            V[(0 * 2 + 2) * 7 + i][col_offset + j] = V_wino[i][j]
            W[(2 * 2 + 0) * 2 + i][col_offset + j] = W_wino[i][j]
    col_offset += 7
    
    # Block 4: M_{2,2,2} for A[0:2,2:4] * B[2:4,2:4] -> C[2:4,0:2]
    for i in range(4):
        for j in range(7):
            U[i * 2 + 2][col_offset + j] += U_wino[i][j]
            V[(2 * 2 + 2) * 7 + i][col_offset + j] = V_wino[i][j]
            W[(2 * 2 + 0) * 2 + i][col_offset + j] += W_wino[i][j]
    col_offset += 7
    
    # Block 5: M_{2,2,2} for A[0:2,0:2] * B[0:2,4:6] -> C[4:6,0:2]
    for i in range(4):
        for j in range(7):
            U[i * 2 + 0][col_offset + j] += U_wino[i][j]
            V[(0 * 2 + 4) * 7 + i][col_offset + j] = V_wino[i][j]
            W[(4 * 2 + 0) * 2 + i][col_offset + j] = W_wino[i][j]
    col_offset += 7
    
    # Block 6: M_{2,2,2} for A[0:2,2:4] * B[2:4,4:6] -> C[4:6,0:2]
    for i in range(4):
        for j in range(7):
            U[i * 2 + 2][col_offset + j] += U_wino[i][j]
            V[(2 * 2 + 4) * 7 + i][col_offset + j] = V_wino[i][j]
            W[(4 * 2 + 0) * 2 + i][col_offset + j] += W_wino[i][j]
    col_offset += 7
    
    # Block 7: M_{2,2,1} for A[0:2,0:2] * B[0:2,6:7] -> C[6:7,0:2]
    for i in range(4):
        for j in range(4):
            U[i * 2 + 0][col_offset + j] += U_wino_2x2x1[i][j]
            V[(0 * 2 + 6) * 7 + i][col_offset + j] = V_wino_2x2x1[i][j]
            W[(6 * 2 + 0) * 2 + i][col_offset + j] = W_wino_2x2x1[i][j]
    col_offset += 4
    
    # Block 8: M_{2,2,1} for A[0:2,2:4] * B[2:4,6:7] -> C[6:7,0:2]
    for i in range(4):
        for j in range(4):
            U[i * 2 + 2][col_offset + j] += U_wino_2x2x1[i][j]
            V[(2 * 2 + 6) * 7 + i][col_offset + j] = V_wino_2x2x1[i][j]
            W[(6 * 2 + 0) * 2 + i][col_offset + j] += W_wino_2x2x1[i][j]
    col_offset += 4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
