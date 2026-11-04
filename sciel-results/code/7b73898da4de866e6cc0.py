import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _build_target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.max(np.abs(T_recon - T))
    return diff < tol, diff

def _schoolbook_decomposition(n, m, p):
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

def _strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2 matrix multiplication: C = A * B
    A indices: 0=A00, 1=A01, 2=A10, 3=A11
    B indices: 0=B00, 1=B01, 2=B10, 3=B11
    C indices: 0=C00, 1=C01, 2=C10, 3=C11
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A10 + A11) * (B00 - B01)
    U[2][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = -1
    W[2][0] = 1; W[3][0] = 1
    
    # M2 = A00 * (B01 + B11)
    U[0][1] = 1
    V[1][1] = 1; V[3][1] = 1
    W[0][1] = 1; W[1][1] = 1
    
    # M3 = (A00 + A01) * (B10 - B00)
    U[0][2] = 1; U[1][2] = 1
    V[2][2] = 1; V[0][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11 * (B10 + B11)
    U[3][3] = 1
    V[2][3] = 1; V[3][3] = 1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A01 + A11) * (B00 + B11)
    U[1][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1; W[2][4] = 1; W[3][4] = 1
    
    # M6 = (A00 - A10) * (B10 + B11)
    U[0][5] = 1; U[2][5] = -1
    V[2][5] = 1; V[3][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7 = (A01 - A11) * (B00 + B10)
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[2][6] = 1
    W[0][6] = 1; W[1][6] = 1; W[2][6] = 1; W[3][6] = 1
    
    return U, V, W

def _rank21_decomposition(n, m, p):
    """
    Rank-21 decomposition:
    - Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (7)
    - Block 2: A[0:2,0:2] * B[0:2,3:5] -> C[0:2,3:5] (7)
    - Block 3: A[0:2,0:2] * B[0:2,2:3] -> C[0:2,2:3] (2)
    - Block 4: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5] (5)
    Total: 21
    """
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    a_map1 = [0, 1, 3, 4] # i*3+j for i in 0,1 and j in 0,1
    b_map1 = [0, 1, 5, 6] # j*5+k for j in 0,1 and k in 0,1
    c_map1 = [0, 1, 2, 3] # k*2+i for k in 0,1 and i in 0,1
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[a_map1[s]][col] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map1[s]][col] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map1[s]][col] += W_s[s][r]
        col += 1
    
    # Block 2: A[0:2,0:2] * B[0:2,3:5] -> C[0:2,3:5]
    a_map2 = [0, 1, 3, 4]
    b_map2 = [3, 4, 8, 9] # j*5+k for j in 0,1 and k in 3,4
    c_map2 = [6, 7, 8, 9] # k*2+i for k in 3,4 and i in 0,1
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[a_map2[s]][col] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map2[s]][col] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map2[s]][col] += W_s[s][r]
        col += 1
        
    # Block 3: A[0:2,0:2] * B[0:2,2:3] -> C[0:2,2:3]
    # Rank 2 decomposition for 2x1x2
    # M1 = (A00+A10)*B02
    # M2 = (A01+A11)*B12
    # Note: B index is j*5+k. j=0, k=2 -> 2. j=1, k=2 -> 7.
    # A index is i*3+j. i=0, j=0 -> 0. i=1, j=0 -> 3. i=0, j=1 -> 1. i=1, j=1 -> 4.
    # C index is k*2+i. k=2. i=0 -> 4. i=1 -> 5.
    
    # Term 1
    U[0][col] = 1 # A[0,0]
    U[3][col] = 1 # A[1,0]
    V[2][col] = 1 # B[0,2]
    W[4][col] = 1 # C[2,0] (k=2, i=0)
    W[5][col] = 1 # C[2,1] (k=2, i=1)
    col += 1
    
    # Term 2
    U[1][col] = 1 # A[0,1]
    U[4][col] = 1 # A[1,1]
    V[7][col] = 1 # B[1,2]
    W[4][col] = 1 # C[2,0]
    W[5][col] = 1 # C[2,1]
    col += 1
    
    # Block 4: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5]
    # Rank 5 decomposition for 2x1x5
    # Term k: A[i,2] * B[2,k] -> C[i,k]
    # A index i*3+2. i=0 -> 2. i=1 -> 5.
    # B index 2*5+k = 10+k.
    # C index k*2+i.
    
    for k in range(5):
        U[2][col] = 1 # A[0,2]
        U[5][col] = 1 # A[1,2]
        V[10+k][col] = 1 # B[2,k]
        W[k*2+0][col] = 1 # C[k,0]
        W[k*2+1][col] = 1 # C[k,1]
        col += 1
        
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # First verify Strassen is correct
    U_s, V_s, W_s = _strassen_2x2x2()
    T_s = np.zeros((4, 4, 4), dtype=np.float64)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                i = a // 2
                j = a % 2
                jj = b // 2
                k = b % 2
                kk = c // 2
                ii = c % 2
                if j == jj and i == ii:
                    T_s[a, b, c] = 1.0
    valid_s, diff_s = _verify_decomposition(U_s, V_s, W_s, T_s)
    
    if not valid_s:
        # Strassen is wrong, fall back to schoolbook
        return _schoolbook_decomposition(n, m, p)
    
    result = _rank21_decomposition(n, m, p)
    valid, diff = _verify_decomposition(result['U'], result['V'], result['W'], T)
    
    if valid:
        return result
    
    # Fallback to schoolbook if 21 fails
    return _schoolbook_decomposition(n, m, p)

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result['U'] else 0}")
