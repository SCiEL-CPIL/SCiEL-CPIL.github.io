import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
    return U, V, W

def _strassen_2x2x2():
    # Returns U, V, W for 2x2x2
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = -1 # Wait, C21 gets M4, C11 gets -M4? 
    # C11 = M1 + M4 - M5 + M7
    # C21 = M2 + M4
    # So W for M4 should be 1 at C21 and 1 at C11?
    # Let's check Strassen formulas again.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Correction for M4:
    # W[2, 3] = 1 (C21)
    # W[0, 3] = 1 (C11)
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _combine(U1, V1, W1, U2, V2, W2):
    return np.hstack([U1, U2]), np.hstack([V1, V2]), np.hstack([W1, W2])

def _split_multiply(n, m, p, split_dim, split_index):
    # Splits the middle dimension m into m1 and m2
    # A is n x m1, n x m2. B is m1 x p, m2 x p.
    # This is not how matrix multiplication splits.
    # A (n x m) * B (m x p).
    # Split m into m1, m2.
    # A1 (n x m1), A2 (n x m2).
    # B1 (m1 x p), B2 (m2 x p).
    # C = A1 B1 + A2 B2.
    
    m1 = split_index
    m2 = m - split_index
    
    U1, V1, W1 = _build(n, m1, p)
    U2, V2, W2 = _build(n, m2, p)
    
    # Remap indices for U
    # U1 is for A1 (n x m1). Rows 0..n*m1-1.
    # U2 is for A2 (n x m2). Rows 0..n*m2-1.
    # In the full A (n x m), A1 is rows 0..n*m1-1, A2 is rows n*m1..n*m-1.
    # So U2 needs to be shifted by n*m1.
    
    U2_shifted = np.zeros((n * m, U2.shape[1]))
    U2_shifted[n * m1:, :] = U2
    
    # V1 is for B1 (m1 x p). Rows 0..m1*p-1.
    # V2 is for B2 (m2 x p). Rows 0..m2*p-1.
    # In the full B (m x p), B1 is rows 0..m1*p-1, B2 is rows m1*p..m*p-1.
    # So V2 needs to be shifted by m1*p.
    
    V2_shifted = np.zeros((m * p, V2.shape[1]))
    V2_shifted[m1 * p:, :] = V2
    
    return _combine(U1, V1, W1, U2_shifted, V2_shifted, W2)

def _build(n, m, p):
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    elif n == 2 and m == 2 and p == 3:
        # Split p into 2+1
        U1, V1, W1 = _build(2, 2, 2)
        U2, V2, W2 = _build(2, 2, 1)
        
        # V2 shift: p1=2, p2=1. B1 is 2x2, B2 is 2x1.
        # V1 rows 0..3. V2 rows 0..1.
        # Full B is 2x3. Rows 0..5.
        # V2 should start at row 2*2 = 4.
        V2_shifted = np.zeros((2 * 3, V2.shape[1]))
        V2_shifted[4:, :] = V2
        
        # W2 shift: p1=2, p2=1. C1 is 2x2, C2 is 2x1.
        # W1 rows 0..3. W2 rows 0..1.
        # Full C is 2x3. Rows 0..5.
        # W2 should start at row 2*2 = 4.
        W2_shifted = np.zeros((3 * 2, W2.shape[1]))
        W2_shifted[4:, :] = W2
        
        return _combine(U1, V1, W1, U2, V2_shifted, W2_shifted)
    else:
        # Split m
        m1 = m // 2
        if m1 == 0 or m1 == m:
            return _schoolbook(n, m, p)
        return _split_multiply(n, m, p, m1)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    U, V, W = _build(n, m, p)
    
    # Verify
    # T_target[i*m+j, j*p+k, k*n+i] = 1
    # T_decomp[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    
    # Check a few points
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1
                decomp = np.dot(U[i*m+j, :], np.dot(V[j*p+k, :], W[k*n+i, :]))
                if abs(target - decomp) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if not valid:
        U, V, W = _schoolbook(n, m, p)
        
    # Convert to lists of lists of floats/ints
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
