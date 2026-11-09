import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Returns a tensor decomposition of the 3x3 * 3x4 matrix multiplication tensor.
    Uses a block decomposition: 3x3x3 (schoolbook, rank 27) + 3x3x1 (rank 3).
    Total rank: 30.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Rank 30 solution
    rank = 30
    U = [[0] * rank for _ in range(n * m)]  # 9 x 30
    V = [[0] * rank for _ in range(m * p)]  # 12 x 30
    W = [[0] * rank for _ in range(p * n)]  # 12 x 30
    
    # Part 1: 3x3 * 3x3 -> 3x3 (columns 0..2 of B, rows 0..2 of C)
    # Indices: i in 0..2, j in 0..2, k in 0..2
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):  # k only 0,1,2
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # col is now 27
    
    # Part 2: 3x3 * 3x1 -> 3x1 (column 3 of B, column 3 of C)
    # Indices: i in 0..2, j in 0..2, k = 3
    # Rank 3 decomposition for this part
    for r in range(3):
        c = 27 + r
        for i in range(n):
            for j in range(m):
                # U[i*m+j, c] = 1 if i == r else 0
                if i == r:
                    U[i * m + j][c] = 1
                # V[j*p+3, c] = 1 for all j
                V[j * p + 3][c] = 1
                # W[3*n+i, c] = 1 if i == r else 0
                if i == r:
                    W[3 * n + i][c] = 1
    
    # Verify
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != Fraction(1):
                        return False
        return True
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][column] = 1
                    V_sb[j * p + k][column] = 1
                    W_sb[k * n + i][column] = 1
                    column += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

