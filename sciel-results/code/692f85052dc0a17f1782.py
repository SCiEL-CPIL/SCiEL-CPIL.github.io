import time

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation: 4 blocks of 2x2x2 (rank 7) + 1 block of 2x2x1 (rank 4)
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 4 Strassen blocks for columns 0-1, 2-3, 4-5, 6-7
    for b in range(4):
        k0 = 2 * b
        
        # M1: (A00 + A11) * (B0k0 + B1k0+1)
        U[0][col] += 1
        U[3][col] += 1
        V[0 * p + k0][col] += 1
        V[1 * p + k0 + 1][col] += 1
        # C0k0 += M1
        W[k0 * n + 0][col] += 1
        # C1k0+1 += M1
        W[(k0 + 1) * n + 1][col] += 1
        col += 1
        
        # M2: (A10 + A11) * B0k0
        U[2][col] += 1
        U[3][col] += 1
        V[0 * p + k0][col] += 1
        # C1k0 += M2
        W[k0 * n + 1][col] += 1
        # C1k0+1 -= M2
        W[(k0 + 1) * n + 1][col] -= 1
        col += 1
        
        # M3: A00 * (B0k0+1 - B1k0)
        U[0][col] += 1
        V[0 * p + k0 + 1][col] += 1
        V[1 * p + k0][col] -= 1
        # C0k0+1 += M3
        W[(k0 + 1) * n + 0][col] += 1
        # C1k0+1 += M3
        W[(k0 + 1) * n + 1][col] += 1
        col += 1
        
        # M4: A11 * (B1k0 - B0k0)
        U[3][col] += 1
        V[1 * p + k0][col] += 1
        V[0 * p + k0][col] -= 1
        # C0k0 += M4
        W[k0 * n + 0][col] += 1
        # C1k0 += M4
        W[k0 * n + 1][col] += 1
        col += 1
        
        # M5: (A00 + A01) * B1k0+1
        U[0][col] += 1
        U[1][col] += 1
        V[1 * p + k0 + 1][col] += 1
        # C0k0 -= M5
        W[k0 * n + 0][col] -= 1
        # C0k0+1 += M5
        W[(k0 + 1) * n + 0][col] += 1
        col += 1
        
        # M6: (A10 - A00) * (B0k0 + B0k0+1)
        U[2][col] += 1
        U[0][col] -= 1
        V[0 * p + k0][col] += 1
        V[0 * p + k0 + 1][col] += 1
        # C1k0+1 += M6
        W[(k0 + 1) * n + 1][col] += 1
        col += 1
        
        # M7: (A01 - A11) * (B1k0 + B1k0+1)
        U[1][col] += 1
        U[3][col] -= 1
        V[1 * p + k0][col] += 1
        V[1 * p + k0 + 1][col] += 1
        # C0k0 += M7
        W[k0 * n + 0][col] += 1
        col += 1
        
    # 1 Schoolbook block for column 8
    k0 = 8
    
    # M1: A00 * B0k0
    U[0][col] += 1
    V[0 * p + k0][col] += 1
    W[k0 * n + 0][col] += 1
    col += 1
    
    # M2: A01 * B1k0
    U[1][col] += 1
    V[1 * p + k0][col] += 1
    W[k0 * n + 0][col] += 1
    col += 1
    
    # M3: A10 * B0k0
    U[2][col] += 1
    V[0 * p + k0][col] += 1
    W[k0 * n + 1][col] += 1
    col += 1
    
    # M4: A11 * B1k0
    U[3][col] += 1
    V[1 * p + k0][col] += 1
    W[k0 * n + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
