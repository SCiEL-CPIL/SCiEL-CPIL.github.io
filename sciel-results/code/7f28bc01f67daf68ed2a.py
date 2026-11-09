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

    def strassen_block(U, V, W, i1, i2, j1, j2, k1, k2, col):
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

    # Build the decomposition using 8 disjoint 2x2x2 Strassen blocks
    # i in {0,1,2,3}, j in {0,1,2,3,4}, k in {0,1,2,3,4}
    # Strassen blocks cover i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3}
    # Standard terms cover the remaining 36 entries
    
    U = [[0] * 92 for _ in range(nm)]
    V = [[0] * 92 for _ in range(mp)]
    W = [[0] * 92 for _ in range(pn)]
    
    blocks = []
    for i_base in [0, 2]:
        for j_base in [0, 2]:
            for k_base in [0, 2]:
                blocks.append((i_base, i_base+1, j_base, j_base+1, k_base, k_base+1))
    
    skip_set = set()
    for i1, i2, j1, j2, k1, k2 in blocks:
        for i in [i1, i2]:
            for j in [j1, j2]:
                for k in [k1, k2]:
                    skip_set.add((i, j, k))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in skip_set:
                    standard_term(U, V, W, i, j, k, col)
                    col += 1
    
    for i1, i2, j1, j2, k1, k2 in blocks:
        col = strassen_block(U, V, W, i1, i2, j1, j2, k1, k2, col)
    
    rank = col
    assert rank == 92, f"Expected 92, got {rank}"
    
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
                    standard_term(U, V, W, i, j, k, col)
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to clean list format with integer entries
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
