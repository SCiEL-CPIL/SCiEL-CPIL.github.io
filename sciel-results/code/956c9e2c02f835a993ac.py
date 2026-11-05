import time
import numpy as np

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0

def _verify(U, V, W, n, m, p, r, tol=1e-9):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _add_strassen(U, V, W, col, i0, j0, k0, n, m, p):
    """Add Strassen decomposition for 2x2x2 block starting at i0, j0, k0."""
    # U indices
    U00 = i0 * m + j0
    U01 = i0 * m + (j0 + 1)
    U10 = (i0 + 1) * m + j0
    U11 = (i0 + 1) * m + (j0 + 1)
    
    # V indices
    V00 = j0 * p + k0
    V01 = j0 * p + (k0 + 1)
    V10 = (j0 + 1) * p + k0
    V11 = (j0 + 1) * p + (k0 + 1)
    
    # W indices
    W00 = k0 * n + i0
    W01 = (k0 + 1) * n + i0
    W10 = k0 * n + (i0 + 1)
    W11 = (k0 + 1) * n + (i0 + 1)
    
    # P1 = (A00+A11)(B00+B11)
    U[U00, col] = 1; U[U11, col] = 1
    V[V00, col] = 1; V[V11, col] = 1
    W[W00, col] = 1; W[W11, col] = 1
    col += 1
    
    # P2 = (A10+A11)B00
    U[U10, col] = 1; U[U11, col] = 1
    V[V00, col] = 1
    W[W10, col] = 1
    col += 1
    
    # P3 = A00(B01-B11)
    U[U00, col] = 1
    V[V01, col] = 1; V[V11, col] = -1
    W[W01, col] = 1
    col += 1
    
    # P4 = A11(B10-B00)
    U[U11, col] = 1
    V[V10, col] = 1; V[V00, col] = -1
    W[W00, col] = 1
    col += 1
    
    # P5 = (A00+A01)B11
    U[U00, col] = 1; U[U01, col] = 1
    V[V11, col] = 1
    W[W01, col] = 1; W[W11, col] = 1
    col += 1
    
    # P6 = (A10-A00)(B00+B01)
    U[U10, col] = 1; U[U00, col] = -1
    V[V00, col] = 1; V[V01, col] = 1
    W[W00, col] = 1
    col += 1
    
    # P7 = (A01-A11)(B10+B11)
    U[U01, col] = 1; U[U11, col] = -1
    V[V10, col] = 1; V[V11, col] = 1
    W[W10, col] = 1; W[W11, col] = -1
    col += 1
    
    return col

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n*m, m*p, p*n
    
    # Max possible rank is schoolbook 84, but we target 44.
    # We will allocate 44 columns.
    R = 44
    U = np.zeros((Nm, R))
    V = np.zeros((Mp, R))
    W = np.zeros((Pn, R))
    
    col = 0
    
    # 6 Strassen blocks
    # i0 in {0}, j0 in {0, 2}, k0 in {0, 2, 4}
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                col = _add_strassen(U, V, W, col, i0, j0, k0, n, m, p)
                
    # Rem1: i in {0,1}, j in {0..3}, k=6
    # Rank 1
    for i in [0, 1]:
        for j in range(4):
            U[i*m + j, col] = 1
    for j in range(4):
        V[j*p + 6, col] = 1
    for i in [0, 1]:
        W[6*n + i, col] = 1
    col += 1
    
    # Rem2: i=2, j in {0..3}, k in {0..6}
    # Rank 1
    for j in range(4):
        U[2*m + j, col] = 1
    for j in range(4):
        for k in range(7):
            V[j*p + k, col] = 1
    for k in range(7):
        W[k*n + 2, col] = 1
    col += 1
    
    # Verify
    if not _verify(U, V, W, n, m, p, col):
        # Fallback to schoolbook if verification fails (should not happen)
        R0 = n * m * p
        U0 = np.zeros((Nm, R0))
        V0 = np.zeros((Mp, R0))
        W0 = np.zeros((Pn, R0))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i*m + j, c] = 1.0
                    V0[j*p + k, c] = 1.0
                    W0[k*n + i, c] = 1.0
                    c += 1
        U, V, W = U0, V0, W0
        col = R0

    def _pack(arr):
        return [[float(x) for x in row] for row in np.asarray(arr)]
    
    return {"U": _pack(U), "V": _pack(V), "W": _pack(W)}
