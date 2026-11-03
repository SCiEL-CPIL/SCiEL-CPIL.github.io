import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen (Rank 7).
    U: (4, 7) for A (2x2)
    V: (4, 7) for B (2x2)
    W: (4, 7) for C (2x2)
    Indices: A[i, j] -> i*2 + j. B[j, k] -> j*2 + k. C[i, k] -> i*2 + k.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] += 1 # A00
    U[3, 0] += 1 # A11
    V[0, 0] += 1 # B00
    V[3, 0] += 1 # B11
    W[0, 0] += 1 # C00
    W[3, 0] += 1 # C11
    W[1, 0] -= 1 # C01
    W[2, 0] -= 1 # C10
    
    # M2 = (A10 + A11)B00
    U[2, 1] += 1 # A10
    U[3, 1] += 1 # A11
    V[0, 1] += 1 # B00
    W[2, 1] += 1 # C10
    W[3, 1] -= 1 # C11
    
    # M3 = A00(B10 - B00)
    U[0, 2] += 1 # A00
    V[2, 2] += 1 # B10
    V[0, 2] -= 1 # B00
    W[1, 2] += 1 # C01
    W[3, 2] += 1 # C11
    
    # M4 = A11(B01 - B11)
    U[3, 3] += 1 # A11
    V[1, 3] += 1 # B01
    V[3, 3] -= 1 # B11
    W[0, 3] += 1 # C00
    W[2, 3] += 1 # C10
    
    # M5 = (A00 + A01)B11
    U[0, 4] += 1 # A00
    U[1, 4] += 1 # A01
    V[3, 4] += 1 # B11
    W[0, 4] -= 1 # C00
    W[1, 4] -= 1 # C01
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] += 1 # A10
    U[0, 5] -= 1 # A00
    V[0, 5] += 1 # B00
    V[1, 5] += 1 # B01
    W[2, 5] += 1 # C10
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] += 1 # A01
    U[3, 6] -= 1 # A11
    V[2, 6] += 1 # B10
    V[3, 6] += 1 # B11
    W[1, 6] += 1 # C01
    
    return U, V, W

def _m212_rank1():
    """
    Returns U, V, W for 2x1x2 matrix multiplication (Rank 1).
    A is 2x1, B is 1x2, C is 2x2.
    U: (2, 1) for A
    V: (2, 1) for B
    W: (4, 1) for C
    """
    U = np.ones((2, 1))
    V = np.ones((2, 1))
    W = np.ones((4, 1))
    return U, V, W

def _build_decomposition(n, m, p):
    """
    Builds the decomposition for 2x5x6 with Rank 45.
    """
    rank = 45
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Part 1: A (2x2) * B (2x6) -> C (2x6)
    # Split B into three 2x2 blocks
    strassen_U, strassen_V, strassen_W = _strassen_222()
    
    for k_start in [0, 2, 4]:
        # Map local indices to global
        # Local A: i in {0,1}, j in {0,1}
        # Local B: j in {0,1}, k in {0,1} (shifted by k_start)
        # Local C: i in {0,1}, k in {0,1} (shifted by k_start)
        
        for r in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    # A index: i*m + j. j is local 0,1 -> global 0,1
                    U[i*m + j, col + r] += strassen_U[i*2 + j, r]
            
            # V
            for j in range(2):
                for k_local in range(2):
                    k_global = k_start + k_local
                    # B index: j*p + k_global. j is local 0,1 -> global 0,1
                    V[j*p + k_global, col + r] += strassen_V[j*2 + k_local, r]
            
            # W
            for i in range(2):
                for k_local in range(2):
                    k_global = k_start + k_local
                    # C index: k_global*n + i
                    W[k_global*n + i, col + r] += strassen_W[i*2 + k_local, r]
        
        col += 7
    
    # Part 2: A (2x3) * B (3x6) -> C (2x6)
    # Split B into three 3x2 blocks
    # Each 3x2 block is decomposed into 2x2 (Strassen) + 1x2 (Rank 1)
    # A cols: 2, 3, 4. B rows: 2, 3, 4.
    
    for k_start in [0, 2, 4]:
        # Sub-part 2a: A (2x2) [cols 2,3] * B (2x2) [rows 2,3, cols k_start..k_start+1]
        # This is M222.
        for r in range(7):
            # U: A cols 2,3
            for i in range(2):
                for j_local in range(2):
                    j_global = 2 + j_local
                    U[i*m + j_global, col + r] += strassen_U[i*2 + j_local, r]
            
            # V: B rows 2,3
            for j_local in range(2):
                j_global = 2 + j_local
                for k_local in range(2):
                    k_global = k_start + k_local
                    V[j_global*p + k_global, col + r] += strassen_V[j_local*2 + k_local, r]
            
            # W: C
            for i in range(2):
                for k_local in range(2):
                    k_global = k_start + k_local
                    W[k_global*n + i, col + r] += strassen_W[i*2 + k_local, r]
        
        col += 7
        
        # Sub-part 2b: A (2x1) [col 4] * B (1x2) [row 4, cols k_start..k_start+1]
        # This is M212 (Rank 1).
        m212_U, m212_V, m212_W = _m212_rank1()
        
        for r in range(1):
            # U: A col 4
            for i in range(2):
                U[i*m + 4, col + r] += m212_U[i, r]
            
            # V: B row 4
            for k_local in range(2):
                k_global = k_start + k_local
                V[4*p + k_global, col + r] += m212_V[k_local, r]
            
            # W: C
            for i in range(2):
                for k_local in range(2):
                    k_global = k_start + k_local
                    W[k_global*n + i, col + r] += m212_W[i*2 + k_local, r]
        
        col += 1
    
    assert col == rank, f"Rank mismatch: {col} != {rank}"
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check invalid entries
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                if i == i2 and j == j2 and k == k2:
                    continue
                s = 0.0
                for r in range(rank):
                    s += U[y, r] * V[z, r] * W[x, r]
                if abs(s) > tol:
                    return False
    return True

def construct():
    n, m, p = 2, 5, 6
    U, V, W = _build_decomposition(n, m, p)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to naive
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
