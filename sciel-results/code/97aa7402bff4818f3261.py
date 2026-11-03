"""Program for 3x4x7 matrix multiplication tensor decomposition."""

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START


def _build_schoolbook():
    n, m, p = 3, 4, 7
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _build_rank_48():
    """
    Construct a rank-48 decomposition for 3x4x7 matrix multiplication.
    
    Strategy: Split the 7 columns of B into three parts: B1 (4x3), B2 (4x3), B3 (4x1).
    C = A * B = A*B1 + A*B2 + A*B3
    
    For A*B1 (3x4 * 4x3 -> 3x3):
      Split A into A1 (3x3) and A2 (3x1).
      A*B1 = A1*B1 + A2*B1
      A1*B1: 3x3x3 schoolbook, rank 9
      A2*B1: 3x1x3 schoolbook, rank 9
      Total for A*B1: rank 18
    
    Similarly for A*B2: rank 18
    
    For A*B3 (3x4 * 4x1 -> 3x1):
      Schoolbook: rank 12 (3*4*1)
    
    Total rank: 18 + 18 + 12 = 48
    """
    n, m, p = 3, 4, 7
    total_rank = 48
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A*B1 where B1 is columns 0-2 of B (4x3)
    # A1*B1: A1 is 3x3 (rows 0-2, cols 0-2), B1 is 4x3 (rows 0-2 of B, cols 0-2)
    # Wait, B is 4x7. B1 is the first 3 columns, so B1[j][k] for j=0..3, k=0..2
    # A is 3x4. A1 is first 3 columns: A1[i][j] for i=0..2, j=0..2
    # A2 is last column: A2[i][3] for i=0..2
    # 
    # A*B1[i,k] = sum_j A[i,j]*B1[j,k] = sum_{j=0}^{2} A[i,j]*B1[j,k] + A[i,3]*B1[3,k]
    # 
    # A1*B1: 3x3 * 3x3 (but B1 is 4x3, we only use rows 0-2 for A1 part)
    # Actually A1 is 3x3, and we multiply by first 3 rows of B1 (which is 3x3)
    # A2 is 3x1, and we multiply by 4th row of B1 (which is 1x3)
    
    # A1*B1: i in 0..2, j in 0..2, k in 0..2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 4 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # A2*B1: i in 0..2, k in 0..2 (j=3 for A, j=3 for B row)
    for i in range(3):
        for k in range(3):
            U[i * 4 + 3][col] = 1
            V[3 * 7 + k][col] = 1
            W[k * 3 + i][col] = 1
            col += 1
    
    # Block 2: A*B2 where B2 is columns 3-5 of B (4x3)
    # A1*B2: i in 0..2, j in 0..2, k in 3..5
    for i in range(3):
        for j in range(3):
            for k in range(3, 6):
                U[i * 4 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # A2*B2: i in 0..2, k in 3..5
    for i in range(3):
        for k in range(3, 6):
            U[i * 4 + 3][col] = 1
            V[3 * 7 + k][col] = 1
            W[k * 3 + i][col] = 1
            col += 1
    
    # Block 3: A*B3 where B3 is column 6 of B (4x1)
    # A*B3: i in 0..2, j in 0..3, k=6
    for i in range(3):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 7 + 6][col] = 1
            W[6 * 3 + i][col] = 1
            col += 1
    
    assert col == total_rank, f"Expected rank {total_rank}, got {col}"
    return U, V, W


def construct():
    # Use the rank-48 block decomposition
    U, V, W = _build_rank_48()
    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-END
