import numpy as np
import time

def construct():
    n, m, p = 2, 2, 8
    deadline = time.time() + 140  # safety margin
    
    # Strassen-based rank-28 decomposition for 2x2x8
    # 4 blocks of 2x2, each using Strassen rank 7
    
    # Strassen 2x2x2 terms: (U, V, W) where U is 4-vector (i*2+j), V is 4-vector (j*2+k), W is 4-vector (k*2+i)
    strassen_terms = [
        # (U, V, W)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # (A00+A11)(B00+B11)
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),   # (A10+A11)B00
        ([1,0,0,0], [0,0,1,-1], [0,1,0,0]),  # A00(B10-B11)
        ([0,0,0,1], [0,1,-1,0], [1,0,0,0]),  # A11(B01-B10)
        ([1,1,0,0], [0,0,0,1], [0,0,0,1]),   # (A00+A01)B11
        ([0,-1,1,0], [1,0,1,0], [0,0,1,0]),  # (A10-A00)(B00+B10)
        ([0,1,0,-1], [0,1,0,1], [0,1,0,0]),  # (A01-A11)(B01+B11)
    ]
    
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 4 x 28
    V = [[0]*rank for _ in range(m*p)]  # 16 x 28
    W = [[0]*rank for _ in range(p*n)]  # 16 x 28
    
    for block in range(4):
        for t in range(7):
            c = block * 7 + t
            u, v, w = strassen_terms[t]
            # U: index i*2+j, u is 4-vector indexed by i*2+j
            for idx in range(4):
                U[idx][c] = u[idx]
            # V: index j*8+k, v is 4-vector for block: v[0]=B[0,2b], v[1]=B[0,2b+1], v[2]=B[1,2b], v[3]=B[1,2b+1]
            # v[j*2+k_local] where k_local in {0,1}, maps to j*8 + (2*block + k_local)
            for j in range(2):
                for k_local in range(2):
                    v_idx = j * 2 + k_local
                    V[j * 8 + (2 * block + k_local)][c] = v[v_idx]
            # W: index k*2+i, w is 4-vector for block: w[0]=C[0,2b], w[1]=C[0,2b+1], w[2]=C[1,2b], w[3]=C[1,2b+1]
            # w[k_local*2+i] maps to (2*block + k_local)*2 + i
            for k_local in range(2):
                for i in range(2):
                    w_idx = k_local * 2 + i
                    W[(2 * block + k_local) * 2 + i][c] = w[w_idx]
    
    # Verify
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(len(U[0])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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
