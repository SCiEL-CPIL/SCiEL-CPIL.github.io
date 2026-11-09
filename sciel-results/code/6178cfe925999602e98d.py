import numpy as np
import time

DIMENSIONS = [3, 5, 6]
_START_TIME = time.time()
_TIMEOUT = 150.0

def _trivial_decomposition(n, m, p):
    """Returns U, V, W for trivial decomposition of M_{n,m,p}."""
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

def _strassen_2x2_decomposition():
    """Returns U, V, W for Strassen decomposition of M_{2,2,2} (Rank 7)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3 = A11(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    
    # M4 = A22(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    
    # W assignments
    # C00 = M1 - M4 + M5 - M7
    W[0][0] = 1
    W[0][3] = -1
    W[0][4] = 1
    W[0][6] = -1
    
    # C10 = M2 + M6
    W[1][1] = 1
    W[1][5] = 1
    
    # C01 = M3 + M7
    W[2][2] = 1
    W[2][6] = 1
    
    # C11 = M1 - M3 + M4 + M6
    W[3][0] = 1
    W[3][2] = -1
    W[3][3] = 1
    W[3][5] = 1
    
    return U, V, W

def _decompose(n, m, p):
    """Recursively decomposes M_{n,m,p} into U, V, W."""
    if time.time() - _START_TIME > _TIMEOUT - 5:
        return _trivial_decomposition(n, m, p)

    # Base cases
    if n == 1 or m == 1 or p == 1:
        return _trivial_decomposition(n, m, p)
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2_decomposition()
    
    # Try splits
    best_rank = n * m * p
    best_U, best_V, best_W = _trivial_decomposition(n, m, p)
    
    # Try splitting n
    for i in range(1, n):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            break
        n1, n2 = i, n - i
        U1, V1, W1 = _decompose(n1, m, p)
        U2, V2, W2 = _decompose(n2, m, p)
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 + r2
        if total_rank < best_rank:
            best_rank = total_rank
            # Concatenate
            U = [row[:] for row in U1]
            # Extend U1 rows with zeros for U2
            for row in U:
                row.extend([0] * r2)
            # Add U2 rows
            for row in U2:
                U.append([0] * r1 + row[:])
            
            V = [row[:] for row in V1]
            for row in V:
                row.extend([0] * r2)
            for row in V2:
                V.append([0] * r1 + row[:])
            
            W = [row[:] for row in W1]
            for row in W:
                row.extend([0] * r2)
            for row in W2:
                W.append([0] * r1 + row[:])
            
            best_U, best_V, best_W = U, V, W

    # Try splitting m
    for j in range(1, m):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            break
        m1, m2 = j, m - j
        U1, V1, W1 = _decompose(n, m1, p)
        U2, V2, W2 = _decompose(n, m2, p)
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 + r2
        if total_rank < best_rank:
            best_rank = total_rank
            U = [row[:] for row in U1]
            for row in U:
                row.extend([0] * r2)
            for row in U2:
                U.append([0] * r1 + row[:])
            
            V = [row[:] for row in V1]
            for row in V:
                row.extend([0] * r2)
            for row in V2:
                V.append([0] * r1 + row[:])
            
            W = [row[:] for row in W1]
            for row in W:
                row.extend([0] * r2)
            for row in W2:
                W.append([0] * r1 + row[:])
            
            best_U, best_V, best_W = U, V, W

    # Try splitting p
    for k in range(1, p):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            break
        p1, p2 = k, p - k
        U1, V1, W1 = _decompose(n, m, p1)
        U2, V2, W2 = _decompose(n, m, p2)
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 + r2
        if total_rank < best_rank:
            best_rank = total_rank
            U = [row[:] for row in U1]
            for row in U:
                row.extend([0] * r2)
            for row in U2:
                U.append([0] * r1 + row[:])
            
            V = [row[:] for row in V1]
            for row in V:
                row.extend([0] * r2)
            for row in V2:
                V.append([0] * r1 + row[:])
            
            W = [row[:] for row in W1]
            for row in W:
                row.extend([0] * r2)
            for row in W2:
                W.append([0] * r1 + row[:])
            
            best_U, best_V, best_W = U, V, W

    return best_U, best_V, best_W

def construct():
    global _START_TIME
    _START_TIME = time.time()
    n, m, p = DIMENSIONS
    U, V, W = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
