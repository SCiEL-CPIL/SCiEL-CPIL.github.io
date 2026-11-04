import time
import numpy as np

_START_TIME = time.time()
DEADLINE = 140.0

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _validate_np(U, V, W, n, m, p):
    U_np = np.asarray(U, dtype=np.float64)
    V_np = np.asarray(V, dtype=np.float64)
    W_np = np.asarray(W, dtype=np.float64)
    rank = U_np.shape[1]
    
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
                
    u_rows = np.array([U_np[idx[0]] for idx in indices])
    v_rows = np.array([V_np[idx[1]] for idx in indices])
    w_rows = np.array([W_np[idx[2]] for idx in indices])
    
    products = u_rows * v_rows * w_rows
    sums = products.sum(axis=1)
    
    if not np.allclose(sums, 1.0, atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _combine(U1, V1, W1, n, m1, p, U2, V2, W2, m2):
    # U1: (n*m1, r1), U2: (n*m2, r2)
    # V1: (m1*p, r1), V2: (m2*p, r2)
    # W1: (p*n, r1), W2: (p*n, r2)
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 + r2
    
    U = [[0] * rank for _ in range(n * (m1 + m2))]
    V = [[0] * rank for _ in range((m1 + m2) * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U
    for i in range(n):
        for j in range(m1):
            for r in range(r1):
                U[i * (m1 + m2) + j][r] = U1[i * m1 + j][r]
        for j in range(m1, m1 + m2):
            for r in range(r1, rank):
                U[i * (m1 + m2) + j][r] = U2[i * m2 + (j - m1)][r - r1]
                
    # Fill V
    for j in range(m1):
        for k in range(p):
            for r in range(r1):
                V[j * p + k][r] = V1[j * p + k][r]
    for j in range(m1, m1 + m2):
        for k in range(p):
            for r in range(r1, rank):
                V[j * p + k][r] = V2[(j - m1) * p + k][r - r1]
                
    # Fill W
    for k in range(p):
        for i in range(n):
            for r in range(r1):
                W[k * n + i][r] = W1[k * n + i][r]
            for r in range(r1, rank):
                W[k * n + i][r] = W2[k * n + i][r - r1]
                
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    # Returns U (4x7), V (4x7), W (4x7)
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    V[0][0] = 1; V[3][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[3][2] = -1
    V[2][3] = 1; V[0][3] = -1
    V[3][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[2][6] = 1; V[3][6] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][2] = 1; W[1][4] = 1
    W[2][1] = 1; W[2][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return {"U": U, "V": V, "W": W}

def get_decomposition(n, m, p):
    if n == 1 and m == 1 and p == 1:
        return {"U": [[1]], "V": [[1]], "W": [[1]]}
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    if m == 1:
        return _schoolbook(n, m, p)
    if n == 1:
        return _schoolbook(n, m, p)
    if p == 1:
        return _schoolbook(n, m, p)
        
    # Split m
    m1 = m // 2
    m2 = m - m1
    
    d1 = get_decomposition(n, m1, p)
    d2 = get_decomposition(n, m2, p)
    
    return _combine(d1["U"], d1["V"], d1["W"], n, m1, p, d2["U"], d2["V"], d2["W"], m2)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = 3, 5, 5
    
    # Try recursive decomposition
    res = get_decomposition(n, m, p)
    
    # Validate
    if _validate_np(res["U"], res["V"], res["W"], n, m, p):
        return res
        
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

if __name__ == "__main__":
    result = construct()
    print(f"Rank achieved: {len(result['U'][0]) if result['U'] else 'unknown'}")
