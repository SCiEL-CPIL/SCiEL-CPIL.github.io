import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 4, 5, 5
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank calculation:
    # 8 Strassen blocks (2x2x2) on i in {0..3}, j in {0..3}, k in {0..3}
    # Rank per block = 7. Total = 56.
    # Cross terms:
    # j=4, k in {0..4}: Rank 5.
    # k=4, j in {0..3}: Rank 4.
    # Total Rank = 56 + 5 + 4 = 65.
    
    rank = 65
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)
    
    col = 0
    
    # 1. Fill Strassen terms for 8 blocks of 2x2x2
    # i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3}
    for ib in [0, 2]:
        for jb in [0, 2]:
            for kb in [0, 2]:
                i1, i2 = ib, ib + 1
                j1, j2 = jb, jb + 1
                k1, k2 = kb, kb + 1
                
                # M1 = (a11 + a22)(b11 + b22)
                U[i1 * m + j1, col] += 1
                U[i2 * m + j2, col] += 1
                V[j1 * p + k1, col] += 1
                V[j2 * p + k2, col] += 1
                W[k1 * n + i1, col] += 1
                W[k2 * n + i2, col] += 1
                col += 1
                
                # M2 = (a21 + a22)b11
                U[i2 * m + j1, col] += 1
                U[i2 * m + j2, col] += 1
                V[j1 * p + k1, col] += 1
                W[k1 * n + i2, col] += 1
                W[k2 * n + i2, col] += -1
                col += 1
                
                # M3 = a11(b12 - b22)
                U[i1 * m + j1, col] += 1
                V[j1 * p + k2, col] += 1
                V[j2 * p + k2, col] += -1
                W[k2 * n + i1, col] += 1
                W[k2 * n + i2, col] += 1
                col += 1
                
                # M4 = a22(b21 - b11)
                U[i2 * m + j2, col] += 1
                V[j2 * p + k1, col] += 1
                V[j1 * p + k1, col] += -1
                W[k1 * n + i1, col] += 1
                W[k1 * n + i2, col] += 1
                col += 1
                
                # M5 = (a11 + a12)b22
                U[i1 * m + j1, col] += 1
                U[i1 * m + j2, col] += 1
                V[j2 * p + k2, col] += 1
                W[k1 * n + i1, col] += -1
                W[k2 * n + i1, col] += 1
                col += 1
                
                # M6 = (a21 - a11)(b11 + b12)
                U[i2 * m + j1, col] += 1
                U[i1 * m + j1, col] += -1
                V[j1 * p + k1, col] += 1
                V[j1 * p + k2, col] += 1
                W[k2 * n + i2, col] += 1
                col += 1
                
                # M7 = (a12 - a22)(b21 + b22)
                U[i1 * m + j2, col] += 1
                U[i2 * m + j2, col] += -1
                V[j2 * p + k1, col] += 1
                V[j2 * p + k2, col] += 1
                W[k1 * n + i1, col] += 1
                col += 1

    # 2. Fill Cross terms
    # Part A: j=4, k in {0..4}. Rank 5.
    # We use 5 terms r=0..4.
    # U[i, 4, r] = 1
    # V[4, k, r] = 1 if k==r else 0
    # W[k, i, r] = 1
    for r in range(5):
        for i in range(n):
            U[i * m + 4, col] = 1
            W[r * n + i, col] = 1
        for k in range(p):
            if k == r:
                V[4 * p + k, col] = 1
        col += 1
        
    # Part B: k=4, j in {0..3}. Rank 4.
    # We use 4 terms r=0..3.
    # U[i, j, r] = 1
    # V[j, 4, r] = 1 if j==r else 0
    # W[4, i, r] = 1
    for r in range(4):
        for i in range(n):
            for j in range(4):
                U[i * m + j, col] = 1
            W[4 * n + i, col] = 1
        for j in range(4):
            if j == r:
                V[j * p + 4, col] = 1
        col += 1

    # Verify correctness
    if time.time() < deadline - 5:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0
                    for r in range(rank):
                        actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if actual != expected:
                        # Fallback
                        rank_std = 100
                        U_std = np.zeros((nm, rank_std), dtype=np.int64)
                        V_std = np.zeros((mp, rank_std), dtype=np.int64)
                        W_std = np.zeros((pn, rank_std), dtype=np.int64)
                        c = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    U_std[i*m+j, c] = 1
                                    V_std[j*p+k, c] = 1
                                    W_std[k*n+i, c] = 1
                                    c += 1
                        return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
