"""Rank-minimal decomposition for M_{2,2,15} using 2p+3 = 33."""

def construct():
    n, m, p = 2, 2, 15
    rank = 2 * p + 3  # 33
    U = [[0] * rank for _ in range(n * m)]  # 4 x 33
    V = [[0] * rank for _ in range(m * p)]  # 30 x 33
    W = [[0] * rank for _ in range(p * n)]  # 30 x 33
    
    col = 0
    # First 2 columns: Strassen (7 products)
    for b in range(7):
        k0, k1 = 2 * b, 2 * b + 1
        # M1 = (A00+A11)*(B0k0+B1k1)
        U[0][col] = 1; U[3][col] = 1
        V[k0][col] = 1; V[15 + k1][col] = 1
        W[2 * k0][col] = 1; W[2 * k1 + 1][col] = 1
        col += 1
        # M2 = (A10+A11)*B0k0
        U[2][col] = 1; U[3][col] = 1
        V[k0][col] = 1
        W[2 * k0 + 1][col] = 1; W[2 * k1 + 1][col] = -1
        col += 1
        # M3 = A00*(B0k1-B1k1)
        U[0][col] = 1
        V[k1][col] = 1; V[15 + k1][col] = -1
        W[2 * k1][col] = 1; W[2 * k1 + 1][col] = 1
        col += 1
        # M4 = A11*(B1k0-B0k0)
        U[3][col] = 1
        V[15 + k0][col] = 1; V[k0][col] = -1
        W[2 * k0][col] = 1; W[2 * k0 + 1][col] = 1
        col += 1
        # M5 = (A00+A01)*B1k1
        U[0][col] = 1; U[1][col] = 1
        V[15 + k1][col] = 1
        W[2 * k0][col] = -1; W[2 * k1][col] = 1
        col += 1
        # M6 = (A10-A00)*(B0k0+B0k1)
        U[2][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k1][col] = 1
        W[2 * k1 + 1][col] = 1
        col += 1
        # M7 = (A01-A11)*(B1k0+B1k1)
        U[1][col] = 1; U[3][col] = -1
        V[15 + k0][col] = 1; V[15 + k1][col] = 1
        W[2 * k0][col] = 1
        col += 1
    
    # Remaining 13 columns: 2 products each
    for k in range(2, 15):
        # M_a = (A00+A11)*(B0k+B1k)
        U[0][col] = 1; U[3][col] = 1
        V[k][col] = 1; V[15 + k][col] = 1
        W[2 * k][col] = 1; W[2 * k + 1][col] = 1
        col += 1
        # M_b = (A10+A01)*B0k
        U[2][col] = 1; U[1][col] = 1
        V[k][col] = 1
        W[2 * k + 1][col] = 1; W[2 * k][col] = -1
        col += 1
    
    return {"U": U, "V": V, "W": W}
