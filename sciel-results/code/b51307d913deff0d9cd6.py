import numpy as np
import time

def _strassen_2x2x2(U, V, W, i_off, j_off, k_off, col, n, m, p):
    """
    Add a 2x2x2 Strassen decomposition at the given offsets.
    Local i' in {0,1}, j' in {0,1}, k' in {0,1}.
    Global i = i_off + i', j = j_off + j', k = k_off + k'.
    U index: i*m + j
    V index: j*p + k
    W index: k*n + i
    """
    c = col
    n2 = n
    m2 = m
    p2 = p

    # M1 = (A00+A11)(B00+B11)
    U[i_off * m2 + j_off, c] = 1
    U[(i_off+1) * m2 + (j_off+1), c] = 1
    V[j_off * p2 + k_off, c] = 1
    V[(j_off+1) * p2 + (k_off+1), c] = 1
    W[k_off * n2 + i_off, c] = 1
    W[(k_off+1) * n2 + (i_off+1), c] = 1
    c += 1

    # M2 = (A10+A11)B00
    U[(i_off+1) * m2 + j_off, c] = 1
    U[(i_off+1) * m2 + (j_off+1), c] = 1
    V[j_off * p2 + k_off, c] = 1
    W[k_off * n2 + (i_off+1), c] = 1
    W[(k_off+1) * n2 + (i_off+1), c] = -1
    c += 1

    # M3 = A00(B11-B01)
    U[i_off * m2 + j_off, c] = 1
    V[j_off * p2 + (k_off+1), c] = 1
    V[(j_off+1) * p2 + (k_off+1), c] = -1
    W[(k_off+1) * n2 + i_off, c] = 1
    W[(k_off+1) * n2 + (i_off+1), c] = 1
    c += 1

    # M4 = A11(B01-B00)
    U[(i_off+1) * m2 + (j_off+1), c] = 1
    V[(j_off+1) * p2 + k_off, c] = 1
    V[j_off * p2 + k_off, c] = -1
    W[k_off * n2 + i_off, c] = 1
    W[k_off * n2 + (i_off+1), c] = 1
    c += 1

    # M5 = (A00+A01)B11
    U[i_off * m2 + j_off, c] = 1
    U[i_off * m2 + (j_off+1), c] = 1
    V[(j_off+1) * p2 + (k_off+1), c] = 1
    W[k_off * n2 + i_off, c] = -1
    W[(k_off+1) * n2 + i_off, c] = 1
    c += 1

    # M6 = (A10-A00)(B00+B01)
    U[(i_off+1) * m2 + j_off, c] = 1
    U[i_off * m2 + j_off, c] = -1
    V[j_off * p2 + k_off, c] = 1
    V[j_off * p2 + (k_off+1), c] = 1
    W[(k_off+1) * n2 + (i_off+1), c] = 1
    c += 1

    # M7 = (A01-A11)(B10+B11)
    U[i_off * m2 + (j_off+1), c] = 1
    U[(i_off+1) * m2 + (j_off+1), c] = -1
    V[(j_off+1) * p2 + k_off, c] = 1
    V[(j_off+1) * p2 + (k_off+1), c] = 1
    W[k_off * n2 + i_off, c] = 1
    c += 1

    return c

def construct():
    start_time = time.time()
    n, m, p = 2, 3, 5
    
    R = 28
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    col = 0
    
    # Block 1: j=0 (2x1x5) schoolbook
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    # Block 2a: j=1,2; k=0,1 (2x2x2) Strassen
    col = _strassen_2x2x2(U, V, W, 0, 1, 0, col, n, m, p)
    
    # Block 2b: j=1,2; k=2,3 (2x2x2) Strassen
    col = _strassen_2x2x2(U, V, W, 0, 1, 2, col, n, m, p)
    
    # Block 2c: j=1,2; k=4 (2x2x1) schoolbook
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
            col += 1
    
    assert col == R, f"Column mismatch: {col} != {R}"
    
    # Deadline guard (not strictly needed for static, but per directive)
    elapsed = time.time() - start_time
    # If somehow this took too long (it won't), we still return the valid result.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
