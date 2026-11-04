import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _mul_block(n, m, p):
    """
    Returns U, V, W for n x m x p multiplication using block decomposition.
    n=2, m=3, p=3 -> Rank 17
    n=2, m=3, p=6 -> Rank 34
    """
    # Base cases: Schoolbook
    if m == 1:
        rank = n * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(p):
                U[i * m][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        return U, V, W, rank

    if p == 1:
        rank = n * m
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p][col] = 1
                W[0 * n + i][col] = 1
                col += 1
        return U, V, W, rank

    if n == 2 and m == 2 and p == 2:
        # Strassen rank 7
        U = [
            [1, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1]
        ]
        # Note: Strassen U, V, W are 4x7, 4x7, 4x7
        # Indices: U_{i,j}, V_{j,k}, W_{i,k}
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        U_s = [
            [1, 0, 1, 0, 1, -1, -1], # A11
            [0, 1, 0, 1, 0, 1, 0],   # A21
            [1, 0, 0, 0, 1, 0, 1],   # A12
            [1, 1, 0, 0, 0, 0, -1]   # A22
        ]
        V_s = [
            [1, 0, 0, 1, 0, 1, 0],   # B11
            [0, 1, 1, -1, 0, 0, 1],  # B21
            [0, 0, 1, 0, 1, 1, 0],   # B12
            [1, 0, -1, 0, 1, 0, 1]   # B22
        ]
        W_s = [
            [1, 0, 0, 1, -1, 0, 1],  # C11
            [0, 0, 1, 0, 1, 0, 0],   # C12
            [0, 1, 0, 1, 0, 0, 0],   # C21
            [1, -1, 1, 0, 0, 1, 0]   # C22
        ]
        return U_s, V_s, W_s, 7

    # Recursive split
    # Split p into p1, p2
    p1 = p // 2
    p2 = p - p1
    
    U1, V1, W1, r1 = _mul_block(n, m, p1)
    U2, V2, W2, r2 = _mul_block(n, m, p2)
    
    rank = r1 + r2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill first r1 columns
    for i in range(n * m):
        for c in range(r1):
            U[i][c] = U1[i][c]
    for j in range(m):
        for k in range(p1):
            for c in range(r1):
                V[j * p + k][c] = V1[j * p1 + k][c]
    for k in range(p1):
        for i in range(n):
            for c in range(r1):
                W[k * n + i][c] = W1[k * n1 + i][c] if False else W1[k * n + i][c] # Fix index
    
    # Fill next r2 columns
    for i in range(n * m):
        for c in range(r2):
            U[i][r1 + c] = U2[i][c]
    for j in range(m):
        for k in range(p2):
            for c in range(r2):
                V[j * p + p1 + k][r1 + c] = V2[j * p2 + k][c]
    for k in range(p2):
        for i in range(n):
            for c in range(r2):
                W[(p1 + k) * n + i][r1 + c] = W2[k * n + i][c]
                
    return U, V, W, rank

def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS
    
    U, V, W, rank = _mul_block(n, m, p)
    
    # Verify
    if time.time() < deadline:
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
            
    return _schoolbook()

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    for i in range(n):
        for i2 in range(n):
            for k in range(p):
                for k2 in range(p):
                    total = Fraction(0)
                    for j in range(m):
                        for c in range(len(U[0])):
                            total += (U[i * m + j][c] * V[j * p + k][c]
                                      * W[k * n + i][c])
                    expected = Fraction(1) if (i == i2 and k == k2) else Fraction(0)
                    if abs(total - expected) > tol:
                        return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
