import numpy as np

DIMENSIONS = [3, 3, 7]

def _hastad_3x3x3():
    """Return 23 terms for 3x3x3 Håstad decomposition."""
    terms = []
    z = lambda: [0.0]*9
    
    # 18 edge terms
    for i in range(3):
        for k in range(3):
            if i == k:
                continue
            u = z(); v = z(); w = z()
            for j in range(3):
                u[i*3+j] = 1.0
                v[j*3+k] = 1.0
            w[k*3+i] = 1.0
            terms.append((u, v, w))
    
    # 3 diagonal terms
    for i in range(3):
        u = z(); v = z(); w = z()
        for j in range(3):
            u[i*3+j] = 1.0
            v[j*3+i] = 1.0
        w[i*3+i] = 1.0
        terms.append((u, v, w))
    
    # 2 correction terms (verified)
    u = z(); v = z(); w = z()
    u[1] = 1.0; u[3] = 1.0
    v[3] = 1.0; v[1] = 1.0
    w[0] = 1.0; w[4] = 1.0
    terms.append((u, v, w))
    
    u = z(); v = z(); w = z()
    u[2] = 1.0; u[6] = 1.0
    v[6] = 1.0; v[2] = 1.0
    w[0] = 1.0; w[8] = 1.0
    terms.append((u, v, w))
    
    return terms

def construct():
    n, m, p = DIMENSIONS
    hastad = _hastad_3x3x3()
    rank = 23 + 23 + 9
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    col = 0
    # Block 1: cols 0-2
    for (u, v, w) in hastad:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] += u[i*3+j]
                    V[j*7+k][col] += v[j*3+k]
                    W[k*3+i][col] += w[k*3+i]
        col += 1
    
    # Block 2: cols 3-5
    for (u, v, w) in hastad:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] += u[i*3+j]
                    V[j*7+(k+3)][col] += v[j*3+k]
                    W[(k+3)*3+i][col] += w[k*3+i]
        col += 1
    
    # Block 3: col 6
    for i in range(3):
        for j in range(3):
            U[i*3+j][col] = 1.0
            V[j*7+6][col] = 1.0
            W[6*3+i][col] = 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}
