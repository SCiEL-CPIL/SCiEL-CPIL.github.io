"""Matrix multiplication 2,4,7 rank-minimal decomposition attempt."""
import numpy as np

DIMENSIONS = [2, 4, 7]

def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 Strassen decomposition."""
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices:
    # U: i*2+j -> 0:00, 1:01, 2:10, 3:11
    # V: j*2+k -> 0:00, 1:01, 2:10, 3:11
    # W: k*2+i -> 0:00, 1:01, 2:10, 3:11
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # c10, c11
    
    # M3 = a00(b10-b11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # c01, c11
    
    # M4 = a11(b01-b00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # c00, c10
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # c00, c01
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # c11
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1 # c00, c11
    
    return U, V, W

def _schoolbook(n, m, p):
    """Returns U, V, W for schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _combine_split_m(n, m, p, D1, D2, m1, m2):
    """Combine decompositions for A=[A1,A2], B=[B1;B2]."""
    U1, V1, W1 = D1
    U2, V2, W2 = D2
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    R = R1 + R2
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Fill U
    # U1 rows: i*m1 + j1 (j1 < m1)
    # U2 rows: i*m2 + j2 (j2 < m2)
    # U rows: i*m + j
    for i in range(n):
        for j in range(m):
            if j < m1:
                U[i * m + j, :R1] = U1[i * m1 + j, :]
            else:
                U[i * m + j, R1:] = U2[i * m2 + (j - m1), :]
                
    # Fill V
    # V1 rows: j1*p + k
    # V2 rows: j2*p + k
    # V rows: j*p + k
    for j in range(m):
        for k in range(p):
            if j < m1:
                V[j * p + k, :R1] = V1[j * p + k, :]
            else:
                V[j * p + k, R1:] = V2[(j - m1) * p + k, :]
                
    # Fill W
    # W = [W1, W2]
    W[:, :R1] = W1
    W[:, R1:] = W2
    
    return U, V, W

def _combine_split_p(n, m, p, D1, D2, p1, p2):
    """Combine decompositions for C=[C1, C2]."""
    U1, V1, W1 = D1
    U2, V2, W2 = D2
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    R = R1 + R2
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # U = [U1, U2]
    U[:, :R1] = U1
    U[:, R1:] = U2
    
    # V = [V1, V2]
    V[:, :R1] = V1
    V[:, R1:] = V2
    
    # Fill W
    # W1 rows: k1*n + i
    # W2 rows: k2*n + i
    # W rows: k*n + i
    for k in range(p):
        for i in range(n):
            if k < p1:
                W[k * n + i, :R1] = W1[k * n + i, :]
            else:
                W[k * n + i, R1:] = W2[(k - p1) * n + i, :]
                
    return U, V, W

def _combine_split_n(n, m, p, D1, D2, n1, n2):
    """Combine decompositions for C=[C1; C2]."""
    U1, V1, W1 = D1
    U2, V2, W2 = D2
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    R = R1 + R2
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Fill U
    # U1 rows: i1*m + j
    # U2 rows: i2*m + j
    # U rows: i*m + j
    for i in range(n):
        for j in range(m):
            if i < n1:
                U[i * m + j, :R1] = U1[i * m + j, :]
            else:
                U[i * m + j, R1:] = U2[(i - n1) * m + j, :]
                
    # V = [V1, V2]
    V[:, :R1] = V1
    V[:, R1:] = V2
    
    # Fill W
    # W1 rows: k*n1 + i1
    # W2 rows: k*n2 + i2
    # W rows: k*n + i
    for k in range(p):
        for i in range(n):
            if i < n1:
                W[k * n + i, :R1] = W1[k * n1 + i, :]
            else:
                W[k * n + i, R1:] = W2[k * n2 + (i - n1), :]
                
    return U, V, W

def _get_decomp(n, m, p):
    # Base case: 2x2x2 uses Strassen
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
        
    # Prefer splitting m if even
    if m % 2 == 0:
        m1 = m // 2
        m2 = m - m1
        D1 = _get_decomp(n, m1, p)
        D2 = _get_decomp(n, m2, p)
        return _combine_split_m(n, m, p, D1, D2, m1, m2)
        
    # Prefer splitting p if even
    if p % 2 == 0:
        p1 = p // 2
        p2 = p - p1
        D1 = _get_decomp(n, m, p1)
        D2 = _get_decomp(n, m, p2)
        return _combine_split_p(n, m, p, D1, D2, p1, p2)
        
    # Prefer splitting n if even
    if n % 2 == 0:
        n1 = n // 2
        n2 = n - n1
        D1 = _get_decomp(n1, m, p)
        D2 = _get_decomp(n2, m, p)
        return _combine_split_n(n, m, p, D1, D2, n1, n2)
        
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _get_decomp(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
