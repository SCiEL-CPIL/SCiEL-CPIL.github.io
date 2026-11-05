import numpy as np

def construct():
    n, m, p = 3, 4, 8
    terms = []

    def add_strassen(i0, j0, k0):
        # Indices for 2x2 block
        a11 = i0 * m + j0
        a12 = i0 * m + j0 + 1
        a21 = (i0 + 1) * m + j0
        a22 = (i0 + 1) * m + j0 + 1
        
        b11 = j0 * p + k0
        b12 = j0 * p + k0 + 1
        b21 = (j0 + 1) * p + k0
        b22 = (j0 + 1) * p + k0 + 1
        
        c11 = k0 * n + i0
        c12 = (k0 + 1) * n + i0
        c21 = k0 * n + i0 + 1
        c22 = (k0 + 1) * n + i0 + 1

        # M1
        u = [0.0] * (n*m); u[a11] = 1; u[a22] = 1
        v = [0.0] * (m*p); v[b11] = 1; v[b22] = 1
        w = [0.0] * (p*n); w[c11] = 1; w[c22] = 1
        terms.append((u, v, w))
        
        # M2
        u = [0.0] * (n*m); u[a21] = 1; u[a22] = 1
        v = [0.0] * (m*p); v[b11] = 1
        w = [0.0] * (p*n); w[c21] = 1; w[c22] = -1
        terms.append((u, v, w))
        
        # M3
        u = [0.0] * (n*m); u[a11] = 1
        v = [0.0] * (m*p); v[b12] = 1; v[b22] = -1
        w = [0.0] * (p*n); w[c12] = 1; w[c22] = 1
        terms.append((u, v, w))
        
        # M4
        u = [0.0] * (n*m); u[a22] = 1
        v = [0.0] * (m*p); v[b21] = 1; v[b11] = -1
        w = [0.0] * (p*n); w[c11] = 1; w[c21] = 1
        terms.append((u, v, w))
        
        # M5
        u = [0.0] * (n*m); u[a11] = 1; u[a12] = 1
        v = [0.0] * (m*p); v[b22] = 1
        w = [0.0] * (p*n); w[c11] = -1; w[c12] = 1
        terms.append((u, v, w))
        
        # M6
        u = [0.0] * (n*m); u[a21] = 1; u[a11] = -1
        v = [0.0] * (m*p); v[b11] = 1; v[b12] = 1
        w = [0.0] * (p*n); w[c22] = 1
        terms.append((u, v, w))
        
        # M7
        u = [0.0] * (n*m); u[a12] = 1; u[a22] = -1
        v = [0.0] * (m*p); v[b21] = 1; v[b22] = 1
        w = [0.0] * (p*n); w[c11] = 1
        terms.append((u, v, w))

    def add_schoolbook_row(i_row, j0, k0, width=4):
        for j in range(width):
            for k in range(4):
                u = [0.0] * (n*m); u[i_row * m + j0 + j] = 1
                v = [0.0] * (m*p); v[(j0 + j) * p + k0 + k] = 1
                w = [0.0] * (p*n); w[(k0 + k) * n + i_row] = 1
                terms.append((u, v, w))

    # 8 Strassen blocks for rows 0-1
    for i0 in [0]: # Only row 0-1 pair
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                add_strassen(i0, j0, k0)
                
    # Wait, the loop above is wrong.
    # Rows 0-1 are i0=0.
    # Cols 0-1 are j0=0. Cols 2-3 are j0=2.
    # Kols 0-1 are k0=0. Kols 2-3 are k0=2. Kols 4-5 are k0=4. Kols 6-7 are k0=6.
    # So 1 * 2 * 4 = 8 blocks. Correct.

    # Schoolbook for row 2
    add_schoolbook_row(2, 0, 0)
    add_schoolbook_row(2, 0, 4)

    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        u, v, w = terms[r]
        for i in range(n * m):
            if u[i] != 0: U[i][r] = u[i]
        for j in range(m * p):
            if v[j] != 0: V[j][r] = v[j]
        for k in range(p * n):
            if w[k] != 0: W[k][r] = w[k]
            
    return {"U": U, "V": V, "W": W}
