"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-13 decomposition for 2x2x5 matrix multiplication.
    Uses Strassen (rank 7) for the 2x2x2 block (B cols 0-1) 
    and schoolbook (rank 6) for the 2x2x3 block (B cols 2-4).
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=5
    rank = 13
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 13)
    V = [[0] * rank for _ in range(m * p)]  # (10, 13)
    W = [[0] * rank for _ in range(p * n)]  # (10, 13)
    
    # Strassen 2x2x2: A(2x2) * B1(2x2) = C1(2x2)
    # U idx: i*2+j, V idx: j*5+k, W idx: k*2+i
    strassen_defs = [
        ([1, 0, 0, 1], [(0, 1), (6, 1)], [(0, 1), (3, 1)]),
        ([0, 0, 1, 1], [(0, 1)], [(2, 1), (3, -1)]),
        ([1, 0, 0, 0], [(1, 1), (6, -1)], [(1, 1), (3, 1)]),
        ([0, 0, 0, 1], [(5, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([1, 1, 0, 0], [(6, 1)], [(0, -1), (1, 1)]),
        ([1, 0, -1, 0], [(0, 1), (1, 1)], [(2, 1)]),
        ([0, 1, 0, -1], [(5, 1), (6, 1)], [(0, -1), (3, 1)]),
    ]
    
    for col, (u_vals, v_updates, w_updates) in enumerate(strassen_defs):
        for row in range(4):
            U[row][col] = u_vals[row]
        for v_idx, coeff in v_updates:
            V[v_idx][col] = coeff
        for w_idx, coeff in w_updates:
            W[w_idx][col] = coeff
    
    # Schoolbook 2x2x3: A(2x2) * B2(2x3) = C2(2x3), B2 cols k=2,3,4
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + (k + 2)][col] = 1
                W[(k + 2) * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
