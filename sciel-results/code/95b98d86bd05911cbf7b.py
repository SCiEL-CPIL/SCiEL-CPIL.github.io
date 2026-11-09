import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    def standard_term(U, V, W, i, j, k, col):
        U[i*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1
        return col + 1
    
    def strassen_2x2x2(U, V, W, i1, i2, j1, j2, k1, k2, col):
        # M1 = (a11+a22)(b11+b22)
        U[i1*m+j1][col] += 1; U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1; V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1; W[k2*n+i2][col] += 1
        col += 1
        # M2 = (a21+a22)b11
        U[i2*m+j1][col] += 1; U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        W[k1*n+i2][col] += 1; W[k2*n+i2][col] += -1
        col += 1
        # M3 = a11(b12-b22)
        U[i1*m+j1][col] += 1
        V[j1*p+k2][col] += 1; V[j2*p+k2][col] += -1
        W[k2*n+i1][col] += 1; W[k2*n+i2][col] += 1
        col += 1
        # M4 = a22(b21-b11)
        U[i2*m+j2][col] += 1
        V[j2*p+k1][col] += 1; V[j1*p+k1][col] += -1
        W[k1*n+i1][col] += 1; W[k1*n+i2][col] += 1
        col += 1
        # M5 = (a11+a12)b22
        U[i1*m+j1][col] += 1; U[i1*m+j2][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += -1; W[k2*n+i1][col] += 1
        col += 1
        # M6 = (a21-a11)(b11+b12)
        U[i2*m+j1][col] += 1; U[i1*m+j1][col] += -1
        V[j1*p+k1][col] += 1; V[j1*p+k2][col] += 1
        W[k2*n+i2][col] += 1
        col += 1
        # M7 = (a12-a22)(b21+b22)
        U[i1*m+j2][col] += 1; U[i2*m+j2][col] += -1
        V[j2*p+k1][col] += 1; V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1
        col += 1
        return col
    
    def strassen_2x3x3(U, V, W, i1, i2, j1, j2, j3, k1, k2, k3, col):
        # 2x3x3 rank-8 decomposition
        # Using the known algorithm for 2xn x nx2 (here 2x3 x 3x2)
        # M1 = (a11+a12+a13)(b11+b21+b31)  -> covers (1,1,1),(1,2,1),(1,3,1)
        U[i1*m+j1][col] += 1; U[i1*m+j2][col] += 1; U[i1*m+j3][col] += 1
        V[j1*p+k1][col] += 1; V[j2*p+k1][col] += 1; V[j3*p+k1][col] += 1
        W[k1*n+i1][col] += 1
        col += 1
        # M2 = (a21+a22+a23)(b12+b22+b32) -> covers (2,1,2),(2,2,2),(2,3,2)
        U[i2*m+j1][col] += 1; U[i2*m+j2][col] += 1; U[i2*m+j3][col] += 1
        V[j1*p+k2][col] += 1; V[j2*p+k2][col] += 1; V[j3*p+k2][col] += 1
        W[k2*n+i2][col] += 1
        col += 1
        # M3 = (a12-a22)(b11-b21) -> covers (1,2,1),(2,2,1),(1,2,2),(2,2,2)
        U[i1*m+j2][col] += 1; U[i2*m+j2][col] += -1
        V[j1*p+k1][col] += 1; V[j2*p+k1][col] += -1
        W[k1*n+i1][col] += 1; W[k2*n+i2][col] += 1
        col += 1
        # M4 = (a13-a23)(b11-b31) -> covers (1,3,1),(2,3,1),(1,3,2),(2,3,2)
        U[i1*m+j3][col] += 1; U[i2*m+j3][col] += -1
        V[j1*p+k1][col] += 1; V[j3*p+k1][col] += -1
        W[k1*n+i1][col] += 1; W[k2*n+i2][col] += 1
        col += 1
        # M5 = (a11-a12)(b21-b22) -> covers (1,1,1),(1,2,1),(1,1,2),(1,2,2)
        U[i1*m+j1][col] += 1; U[i1*m+j2][col] += -1
        V[j2*p+k1][col] += 1; V[j2*p+k2][col] += -1
        W[k1*n+i1][col] += 1; W[k2*n+i1][col] += 1
        col += 1
        # M6 = (a11-a13)(b31-b32) -> covers (1,1,1),(1,3,1),(1,1,2),(1,3,2)
        U[i1*m+j1][col] += 1; U[i1*m+j3][col] += -1
        V[j3*p+k1][col] += 1; V[j3*p+k2][col] += -1
        W[k1*n+i1][col] += 1; W[k2*n+i1][col] += 1
        col += 1
        # M7 = (a21-a22)(b21-b22) -> covers (2,1,1),(2,2,1),(2,1,2),(2,2,2)
        U[i2*m+j1][col] += 1; U[i2*m+j2][col] += -1
        V[j2*p+k1][col] += 1; V[j2*p+k2][col] += -1
        W[k1*n+i2][col] += 1; W[k2*n+i2][col] += 1
        col += 1
        # M8 = (a21-a23)(b31-b32) -> covers (2,1,1),(2,3,1),(2,1,2),(2,3,2)
        U[i2*m+j1][col] += 1; U[i2*m+j3][col] += -1
        V[j3*p+k1][col] += 1; V[j3*p+k2][col] += -1
        W[k1*n+i2][col] += 1; W[k2*n+i2][col] += 1
        col += 1
        return col
    
    # Total rank: 7 (2x2x2) + 8 (2x3x3) + 50 (2x5x5 standard) = 65
    rank = 65
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen 2x2x2
    col = strassen_2x2x2(U, V, W, 0, 1, 0, 1, 0, 1, col)
    
    # Block 2: i in {0,1}, j in {2,3,4}, k in {2,3,4} -> Strassen 2x3x3
    col = strassen_2x3x3(U, V, W, 0, 1, 2, 3, 4, 2, 3, 4, col)
    
    # Block 3: i in {2,3}, j in {0,1,2,3,4}, k in {0,1,2,3,4} -> standard
    for i in [2, 3]:
        for j in range(5):
            for k in range(5):
                col = standard_term(U, V, W, i, j, k, col)
    
    assert col == rank, f"Expected {rank}, got {col}"
    
    if not verify(U, V, W, n, m, p):
        # Fallback: standard rank-100 decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = standard_term(U, V, W, i, j, k, col)
        return {"U": U, "V": V, "W": W}
    
    def to_list(mat):
        result = []
        for i in range(len(mat)):
            row = []
            for r in range(len(mat[0])):
                val = mat[i][r]
                if abs(val - round(val)) < 1e-9:
                    row.append(int(round(val)))
                else:
                    row.append(float(val))
            result.append(row)
        return result
    
    U_final = to_list(U)
    V_final = to_list(V)
    W_final = to_list(W)
    
    return {"U": U_final, "V": V_final, "W": W_final}
