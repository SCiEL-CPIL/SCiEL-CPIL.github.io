import time

def construct():
    n, m, p = 2, 2, 9
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for pairs of columns
    # Pairs: (0,1), (2,3), (4,5), (6,7)
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for k1, k2 in pairs:
        # M1 = (A00 + A11) * (B0k1 + B1k2)
        # U: A00=1, A11=1
        U[0][col] = 1
        U[3][col] = 1
        # V: B0k1=1, B1k2=1
        V[k1][col] = 1
        V[9 + k2][col] = 1
        # W: C0k1 += M1, C1k2 += M1
        W[k1 * 2 + 0][col] = 1
        W[k2 * 2 + 1][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B0k1
        U[2][col] = 1
        U[3][col] = 1
        V[k1][col] = 1
        # W: C1k1 += M2
        W[k1 * 2 + 1][col] = 1
        col += 1
        
        # M3 = A00 * (B0k2 - B1k2)
        U[0][col] = 1
        V[9 + k2][col] = -1
        V[k2][col] = 1
        # W: C0k2 += M3
        W[k2 * 2 + 0][col] = 1
        col += 1
        
        # M4 = A11 * (B1k1 - B0k1)
        U[3][col] = 1
        V[k1][col] = -1
        V[9 + k1][col] = 1
        # W: C0k1 += M4, C1k1 += M4
        W[k1 * 2 + 0][col] = 1
        W[k1 * 2 + 1][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B1k2
        U[0][col] = 1
        U[1][col] = 1
        V[9 + k2][col] = 1
        # W: C0k1 -= M5, C0k2 += M5
        W[k1 * 2 + 0][col] = -1
        W[k2 * 2 + 0][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k1 + B0k2)
        U[0][col] = -1
        U[2][col] = 1
        V[k1][col] = 1
        V[k2][col] = 1
        # W: C1k2 += M6
        W[k2 * 2 + 1][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k1 + B1k2)
        U[1][col] = 1
        U[3][col] = -1
        V[9 + k1][col] = 1
        V[9 + k2][col] = 1
        # W: C0k1 += M7, C1k2 += M7
        W[k1 * 2 + 0][col] = 1
        W[k2 * 2 + 1][col] = 1
        col += 1

    # Schoolbook for single column 8
    k = 8
    # M8 = A00 * B0k
    U[0][col] = 1
    V[k][col] = 1
    W[k * 2 + 0][col] = 1
    col += 1
    
    # M9 = A01 * B1k
    U[1][col] = 1
    V[9 + k][col] = 1
    W[k * 2 + 0][col] = 1
    col += 1
    
    # M10 = A10 * B0k
    U[2][col] = 1
    V[k][col] = 1
    W[k * 2 + 1][col] = 1
    col += 1
    
    # M11 = A11 * B1k
    U[3][col] = 1
    V[9 + k][col] = 1
    W[k * 2 + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
