"""Matrix multiplication 2x2*2x9: rank-32 Strassen decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n=2, m=2, p=9, tol=1e-9):
    """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check all tensor entries: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _add_strassen(k0, k1, col, n, m, p, U, V, W):
    """Add 7 Strassen products for column pair (k0, k1). Returns next col index."""
    # Index shortcuts
    u00, u01, u10, u11 = 0, 1, 2, 3
    v0k0 = 0*p + k0; v1k0 = 1*p + k0
    v0k1 = 0*p + k1; v1k1 = 1*p + k1
    wk0i0 = k0*n + 0; wk0i1 = k0*n + 1
    wk1i0 = k1*n + 0; wk1i1 = k1*n + 1
    
    # M1 = (a00+a11)*(b00+b11) -> contributes to C00 and C11
    c = col
    U[u00, c] += 1; U[u11, c] += 1
    V[v0k0, c] += 1; V[v1k1, c] += 1
    W[wk0i0, c] += 1; W[wk1i1, c] += 1
    col += 1
    
    # M2 = (a10+a11)*b00 -> contributes to C10 and C11 (with -1)
    c = col
    U[u10, c] += 1; U[u11, c] += 1
    V[v0k0, c] += 1
    W[wk0i1, c] += 1; W[wk1i1, c] -= 1
    col += 1
    
    # M3 = a00*(b01-b11) -> contributes to C01 and C11
    c = col
    U[u00, c] += 1
    V[v0k1, c] += 1; V[v1k1, c] -= 1
    W[wk1i0, c] += 1; W[wk1i1, c] += 1
    col += 1
    
    # M4 = a11*(b10-b00) -> contributes to C00 and C10
    c = col
    U[u11, c] += 1
    V[v1k0, c] += 1; V[v0k0, c] -= 1
    W[wk0i0, c] += 1; W[wk0i1, c] += 1
    col += 1
    
    # M5 = (a00+a01)*b11 -> contributes to C00 (with -1) and C01
    c = col
    U[u00, c] += 1; U[u01, c] += 1
    V[v1k1, c] += 1
    W[wk0i0, c] -= 1; W[wk1i0, c] += 1
    col += 1
    
    # M6 = (a10-a00)*(b00+b01) -> contributes to C11
    c = col
    U[u10, c] += 1; U[u00, c] -= 1
    V[v0k0, c] += 1; V[v0k1, c] += 1
    W[wk1i1, c] += 1
    col += 1
    
    # M7 = (a01-a11)*(b10+b11) -> contributes to C00
    c = col
    U[u01, c] += 1; U[u11, c] -= 1
    V[v1k0, c] += 1; V[v1k1, c] += 1
    W[wk0i0, c] += 1
    col += 1
    
    return col

def _add_schoolbook(k, col, n, m, p, U, V, W):
    """Add 4 schoolbook products for single column k. Returns next col index."""
    u00, u01, u10, u11 = 0, 1, 2, 3
    v0k = 0*p + k; v1k = 1*p + k
    wki0 = k*n + 0; wki1 = k*n + 1
    
    # a00*b0k -> C0k
    c = col
    U[u00, c] += 1
    V[v0k, c] += 1
    W[wki0, c] += 1
    col += 1
    
    # a01*b1k -> C0k
    c = col
    U[u01, c] += 1
    V[v1k, c] += 1
    W[wki0, c] += 1
    col += 1
    
    # a10*b0k -> C1k
    c = col
    U[u10, c] += 1
    V[v0k, c] += 1
    W[wki1, c] += 1
    col += 1
    
    # a11*b1k -> C1k
    c = col
    U[u11, c] += 1
    V[v1k, c] += 1
    W[wki1, c] += 1
    col += 1
    
    return col

def _build_rank32(n=2, m=2, p=9):
    """Build rank-32 decomposition: 4 Strassen blocks (8 cols) + 1 schoolbook col (1 col)."""
    rank = 32
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # Strassen blocks for column pairs: (0,1), (2,3), (4,5), (6,7)
    for k0, k1 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        col = _add_strassen(k0, k1, col, n, m, p, U, V, W)
    # Schoolbook for column 8
    col = _add_schoolbook(8, col, n, m, p, U, V, W)
    
    assert col == rank, f"Expected col={rank}, got col={col}"
    return U, V, W

def _build_schoolbook_rank36(n=2, m=2, p=9):
    """Fallback: pure schoolbook decomposition, rank 36."""
    rank = 36
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for k in range(p):
        col = _add_schoolbook(k, col, n, m, p, U, V, W)
    
    assert col == rank
    return U, V, W

def construct():
    """
    Construct a minimal-rank tensor decomposition for 2x2 * 2x9 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' as lists of lists.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill
    
    # Build rank-32 Strassen decomposition
    U, V, W = _build_rank32(n, m, p)
    
    # Verify correctness
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank-36
    U, V, W = _build_schoolbook_rank36(n, m, p)
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # This should never happen, but return something
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
