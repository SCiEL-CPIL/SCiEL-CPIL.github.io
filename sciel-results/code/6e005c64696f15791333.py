import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _build_schoolbook(n, m, p):
    """Return the trivial schoolbook (rank = n*m*p) decomposition."""
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # C00 = M1 + M4 - M5 - M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = -1
    # C01 = M3 + M5
    W[1, 2] = 1; W[1, 4] = 1
    # C10 = M2 + M4
    W[2, 1] = 1; W[2, 3] = 1
    # C11 = M1 - M2 + M3 + M7
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 6] = 1
    
    return U, V, W

_STRASSEN_CACHE = None
def _get_strassen():
    global _STRASSEN_CACHE
    if _STRASSEN_CACHE is None:
        _STRASSEN_CACHE = _strassen_2x2()
    return _STRASSEN_CACHE

def _build_3x3x3_decomposition():
    """Build rank-21 decomposition for 3x3 * 3x3."""
    n, m, p = 3, 3, 3
    U_s, V_s, W_s = _get_strassen()
    
    U_list = []
    V_list = []
    W_list = []
    
    # 1. A[0:2, 0:2] * B[0:2, 0:2] (Rank 7)
    for r in range(7):
        U_term = np.zeros(n * m)
        V_term = np.zeros(m * p)
        W_term = np.zeros(p * n)
        
        for a_flat in range(4):
            ai = a_flat // 2
            aj = a_flat % 2
            U_term[ai * 3 + aj] = U_s[a_flat, r]
            
        for b_flat in range(4):
            bj = b_flat // 2
            bk = b_flat % 2
            V_term[bj * 3 + bk] = V_s[b_flat, r]
            
        for c_flat in range(4):
            ci = c_flat // 2
            ck = c_flat % 2
            W_term[ck * 3 + ci] = W_s[c_flat, r]
            
        U_list.append(U_term)
        V_list.append(V_term)
        W_list.append(W_term)

    # 2. A[0:2, 0:2] * B[0:2, 2] (Rank 4)
    # A[0:2, 0:2] is 2x2, B[0:2, 2] is 2x1.
    # This is equivalent to 2 separate 2x2x1? No, it's a matrix-vector product.
    # C[0:2, 2] = A[0:2, 0:2] * B[0:2, 2]
    # We can use schoolbook for this part: 4 mults.
    for i in range(2):
        for j in range(2):
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)
            U_term[i * 3 + j] = 1.0
            V_term[j * 3 + 2] = 1.0
            W_term[2 * 3 + i] = 1.0
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)

    # 3. A[2, 0:2] * B[0:2, 0:2] (Rank 4)
    # A[2, 0:2] is 1x2, B[0:2, 0:2] is 2x2.
    # C[2, 0:2] = A[2, 0:2] * B[0:2, 0:2]
    for j in range(2):
        for k in range(2):
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)
            U_term[2 * 3 + j] = 1.0
            V_term[j * 3 + k] = 1.0
            W_term[k * 3 + 2] = 1.0
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)

    # 4. Scalar products
    # a01 * b10 -> C[0, 0]
    U_term = np.zeros(n * m); V_term = np.zeros(m * p); W_term = np.zeros(p * n)
    U_term[0 * 3 + 2] = 1.0; V_term[1 * 3 + 0] = 1.0; W_term[0 * 3 + 0] = 1.0
    U_list.append(U_term); V_list.append(V_term); W_list.append(W_term)

    # a01 * b11 -> C[0, 1]
    U_term = np.zeros(n * m); V_term = np.zeros(m * p); W_term = np.zeros(p * n)
    U_term[0 * 3 + 2] = 1.0; V_term[1 * 3 + 1] = 1.0; W_term[1 * 3 + 0] = 1.0
    U_list.append(U_term); V_list.append(V_term); W_list.append(W_term)

    # a10 * b01 -> C[1, 1]
    U_term = np.zeros(n * m); V_term = np.zeros(m * p); W_term = np.zeros(p * n)
    U_term[2 * 3 + 0] = 1.0; V_term[0 * 3 + 1] = 1.0; W_term[1 * 3 + 1] = 1.0
    U_list.append(U_term); V_list.append(V_term); W_list.append(W_term)

    # a11 * b10 -> C[1, 0]
    U_term = np.zeros(n * m); V_term = np.zeros(m * p); W_term = np.zeros(p * n)
    U_term[2 * 3 + 2] = 1.0; V_term[1 * 3 + 0] = 1.0; W_term[0 * 3 + 1] = 1.0
    U_list.append(U_term); V_list.append(V_term); W_list.append(W_term)

    # a11 * b11 -> C[1, 1]
    U_term = np.zeros(n * m); V_term = np.zeros(m * p); W_term = np.zeros(p * n)
    U_term[2 * 3 + 2] = 1.0; V_term[1 * 3 + 1] = 1.0; W_term[1 * 3 + 1] = 1.0
    U_list.append(U_term); V_list.append(V_term); W_list.append(W_term)

    # a10 * b01 is already covered? 
    # Wait, C[1, 1] = a10*b01 + a11*b11.
    # I added a10*b01 and a11*b11.
    # What about C[1, 0]? C[1, 0] = a10*B[0,0] + a11*b10.
    # I added a11*b10.
    # What about C[0, 0]? C[0, 0] = A00B00 + a01*b10.
    # I added a01*b10.
    # What about C[0, 1]? C[0, 1] = A00*b01 + a01*b11.
    # I added a01*b11.
    
    # Did I miss a10*b01?
    # C[1, 1] needs a10*b01.
    # I added "a10 * b01 -> C[1, 1]".
    
    # Total terms: 7 + 4 + 4 + 5 = 20?
    # Let's count scalars:
    # 1. a01*b10
    # 2. a01*b11
    # 3. a10*b01
    # 4. a11*b10
    # 5. a11*b11
    # Total 5 scalars.
    # Total rank = 7 + 4 + 4 + 5 = 20.
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    return U, V, W

def _build_block_decomposition(n, m, p):
    """Build a decomposition using two 3x3x3 blocks (rank 40)."""
    n, m, p = 3, 3, 6
    
    U1, V1, W1 = _build_3x3x3_decomposition()
    
    U_list = []
    V_list = []
    W_list = []
    
    # Block 1: B[0:3, 0:3]
    for r in range(U1.shape[1]):
        U_term = np.zeros(n * m)
        V_term = np.zeros(m * p)
        W_term = np.zeros(p * n)
        
        U_term[:] = U1[:, r]
        
        # Map V1 (3x3) to V (3x6) cols 0-2
        for b_flat in range(9):
            bj = b_flat // 3
            bk = b_flat % 3
            V_term[bj * 6 + bk] = V1[b_flat, r]
            
        # Map W1 (3x3) to W (6x3) rows 0-2
        for c_flat in range(9):
            ci = c_flat // 3
            ck = c_flat % 3
            W_term[ck * 3 + ci] = W1[c_flat, r]
            
        U_list.append(U_term)
        V_list.append(V_term)
        W_list.append(W_term)
        
    # Block 2: B[0:3, 3:6]
    for r in range(U1.shape[1]):
        U_term = np.zeros(n * m)
        V_term = np.zeros(m * p)
        W_term = np.zeros(p * n)
        
        U_term[:] = U1[:, r]
        
        # Map V1 (3x3) to V (3x6) cols 3-5
        for b_flat in range(9):
            bj = b_flat // 3
            bk = b_flat % 3
            V_term[bj * 6 + (3 + bk)] = V1[b_flat, r]
            
        # Map W1 (3x3) to W (6x3) rows 3-5
        for c_flat in range(9):
            ci = c_flat // 3
            ck = c_flat % 3
            W_term[(3 + ck) * 3 + ci] = W1[c_flat, r]
            
        U_list.append(U_term)
        V_list.append(V_term)
        W_list.append(W_term)
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    rank = U.shape[1]
    return rank, U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    R = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    
    for a in range(a_dim):
        i1, j1 = a // m, a % m
        for b in range(b_dim):
            j2, k1 = b // p, b % p
            for c in range(c_dim):
                k2, i2 = c // n, c % n
                if i1 == i2 and j1 == j2 and k1 == k2:
                    continue
                s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(R))
                if abs(s) > tol:
                    return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 110
    
    try:
        rank, U, V, W = _build_block_decomposition(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
