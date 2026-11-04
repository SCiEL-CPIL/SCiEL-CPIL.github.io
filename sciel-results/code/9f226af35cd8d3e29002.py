"""
Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
Strategy: Split A (4x5) into A1 (4x4) and A2 (4x1), B (5x6) into B1 (4x6) and B2 (1x6).
C = A1*B1 + A2*B2.
Use Strassen-like decomposition for A1*B1 (4x4 * 4x6) and trivial for A2*B2.
Total rank: 49 + 24 = 73.
"""

def construct():
    n, m, p = 4, 5, 6
    
    # Split: A1 is 4x4, A2 is 4x1; B1 is 4x6, B2 is 1x6
    n1, n2 = 4, 1  # n = n1 + n2
    m1, m2 = 4, 1  # m = m1 + m2
    p1, p2 = 6, 6  # p = p1 + p2 (B1 and B2 both have 6 columns)
    
    # Total rank
    rank1 = 49  # Strassen for 4x4 * 4x4, but here it's 4x4 * 4x6, need to adjust
    # Actually, let's use a simpler approach: block decomposition
    # A1*B1: 4x4 * 4x6. We can use 16 terms (schoolbook) or Strassen.
    # For 4x4 * 4x4, Strassen is 49. For 4x4 * 4x6, we can treat it as 4x4 * 4x4 + 4x4 * 4x2.
    # Or just use 16 terms for A1*B1 and 24 for A2*B2, total 40.
    # Wait, 16 + 24 = 40 is better than 73.
    # Let's use schoolbook for A1*B1: rank = 4*4 = 16
    # A2*B2: rank = 4*1*6 = 24
    # Total: 40
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]  # 20 rows
    V = [[0] * rank for _ in range(m * p)]  # 30 rows
    W = [[0] * rank for _ in range(p * n)]  # 24 rows
    
    col = 0
    # A1*B1: i in 0..3, j in 0..3, k in 0..5
    # U[i*5 + j, col] = 1
    # V[j*6 + k, col] = 1
    # W[k*4 + i, col] = 1
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # A2*B2: i in 0..3, j=4, k in 0..5
    # U[i*5 + 4, col] = 1
    # V[4*6 + k, col] = 1
    # W[k*4 + i, col] = 1
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U[i * m + (m1 + j)][col] = 1
                V[(m1 + j) * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
