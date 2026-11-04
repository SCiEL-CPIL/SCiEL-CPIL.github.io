import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen terms for A' B'
        # M1 = (A11 + A22)(B11 + B22)
        U[0*m + 0][col] = 1; U[1*m + 1][col] = 1
        V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        
        # M2 = (A21 + A22)B11
        U[1*m + 0][col] = 1; U[1*m + 1][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
        col += 1
        
        # M3 = A11(B21 - B22)
        U[0*m + 0][col] = 1
        V[1*p + k0][col] = 1; V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        
        # M4 = A22(B21 - B11)
        U[1*m + 1][col] = 1
        V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
        col += 1
        
        # M5 = (A11 + A12)B22
        U[0*m + 0][col] = 1; U[0*m + 1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
        col += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[1*m + 0][col] = 1; U[0*m + 0][col] = -1
        V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[0*m + 1][col] = 1; U[1*m + 1][col] = -1
        V[1*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1
        
        # Outer product terms
        # P1 = A13 B31 -> C11
        U[0*m + 2][col] = 1
        V[2*p + k0][col] = 1
        W[k0*n + 0][col] = 1
        col += 1
        
        # P2 = A13 B32 -> C12
        U[0*m + 2][col] = 1
        V[2*p + k1][col] = 1
        W[k1*n + 0][col] = 1
        col += 1
        
        # P3 = A23 B31 -> C21
        U[1*m + 2][col] = 1
        V[2*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        
        # P4 = A23 B32 -> C22
        U[1*m + 2][col] = 1
        V[2*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
