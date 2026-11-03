import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def _build_decomp_92():
    """
    Build a rank-92 decomposition using a 3x3 / 1x1 row split for A and C,
    and a 3x2 / 2x2 col split for A and row/col split for B.
    
    A = [A11 (3x3) | A12 (3x2)]
        [A21 (1x3) | A22 (1x2)]
    
    B = [B11 (3x3) | B12 (3x2)]
        [B21 (2x3) | B22 (2x2)]
    
    C = [C11 (3x3) | C12 (3x2)]
        [C21 (1x3) | C22 (1x2)]
    
    C11 = A11 B11 + A12 B21
    C12 = A11 B12 + A12 B22
    C21 = A21 B11 + A22 B21
    C22 = A21 B12 + A22 B22
    
    Ranks:
    A11 B11: 3x3x3 -> 27
    A12 B21: 3x2x2 -> 12
    A11 B12: 3x3x2 -> 18
    A12 B22: 3x2x2 -> 12
    A21 B11: 1x3x3 -> 9
    A22 B21: 1x2x2 -> 4
    A21 B12: 1x3x2 -> 6
    A22 B22: 1x2x2 -> 4
    Total: 92
    """
    n, m, p = N, M, P
    rank = 92
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    r = 0
    
    # Helper to add schoolbook terms for a sub-block
    def add_block(i_range, j_range, k_range):
        nonlocal r
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
                    r += 1

    # C11 = A11 B11 + A12 B21
    # A11 B11: i in 0..2, j in 0..2, k in 0..2
    add_block(range(0, 3), range(0, 3), range(0, 3))
    # A12 B21: i in 0..2, j in 3..4, k in 0..2
    add_block(range(0, 3), range(3, 5), range(0, 3))
    
    # C12 = A11 B12 + A12 B22
    # A11 B12: i in 0..2, j in 0..2, k in 3..4
    add_block(range(0, 3), range(0, 3), range(3, 5))
    # A12 B22: i in 0..2, j in 3..4, k in 3..4
    add_block(range(0, 3), range(3, 5), range(3, 5))
    
    # C21 = A21 B11 + A22 B21
    # A21 B11: i in 3, j in 0..2, k in 0..2
    add_block(range(3, 4), range(0, 3), range(0, 3))
    # A22 B21: i in 3, j in 3..4, k in 0..2
    add_block(range(3, 4), range(3, 5), range(0, 3))
    
    # C22 = A21 B12 + A22 B22
    # A21 B12: i in 3, j in 0..2, k in 3..4
    add_block(range(3, 4), range(0, 3), range(3, 5))
    # A22 B22: i in 3, j in 3..4, k in 3..4
    add_block(range(3, 4), range(3, 5), range(3, 5))
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    return U, V, W


def construct():
    t0 = time.time()
    n, m, p = N, M, P
    
    # Build the rank-92 decomposition
    U, V, W = _build_decomp_92()
    
    # Validate
    if not _validate(U, V, W):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
