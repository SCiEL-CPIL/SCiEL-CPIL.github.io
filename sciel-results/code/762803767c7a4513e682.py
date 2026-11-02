import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def _validate(U, V, W, r):
    """Validate decomposition with random test matrices."""
    n, m, p = DIMENSIONS
    np.random.seed(12345)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_exp = A @ B
    C_comp = np.zeros((n, p))
    for c in range(r):
        a_c = sum(U[i*m+j][c]*A[i,j] for i in range(n) for j in range(m))
        b_c = sum(V[j*p+k][c]*B[j,k] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_comp[i,k] += W[k*n+i][c]*a_c*b_c
    return np.allclose(C_comp, C_exp, atol=1e-9)

def construct():
    """
    Returns a tensor decomposition for 2x5 x 5x5 matrix multiplication.
    Uses Strassen for 2x2x2 blocks and schoolbook for remaining parts.
    Rank 46.
    """
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 135

    terms = []

    def add_strassen(i_start, j_start, k_start):
        # Adds 7 terms for 2x2x2 block at (i_start, j_start, k_start)
        # i in {i_start, i_start+1}
        # j in {j_start, j_start+1}
        # k in {k_start, k_start+1}
        
        # M1
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M2
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))
        
        # M3
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M4
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))
        
        # M5
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))
        
        # M6
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M7
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))

    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        # Adds (i_end-i_start)*(j_end-j_start)*(k_end-k_start) terms
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))

    # Decomposition plan:
    # n=2 is kept as a single block of size 2.
    # m=5 is split into 2, 2, 1.
    # p=5 is split into 2, 2, 1.
    
    # Block (i:0-1, j:0-1, k:0-1) -> 2x2x2 -> Strassen
    add_strassen(0, 0, 0)
    
    # Block (i:0-1, j:0-1, k:2-3) -> 2x2x2 -> Strassen
    add_strassen(0, 0, 2)
    
    # Block (i:0-1, j:0-1, k:4-4) -> 2x2x1 -> Schoolbook
    add_schoolbook(0, 2, 0, 2, 4, 5)
    
    # Block (i:0-1, j:2-3, k:0-1) -> 2x2x2 -> Strassen
    add_strassen(0, 2, 0)
    
    # Block (i:0-1, j:2-3, k:2-3) -> 2x2x2 -> Strassen
    add_strassen(0, 2, 2)
    
    # Block (i:0-1, j:2-3, k:4-4) -> 2x2x1 -> Schoolbook
    add_schoolbook(0, 2, 2, 4, 4, 5)
    
    # Block (i:0-1, j:4-4, k:0-1) -> 2x1x2 -> Schoolbook
    add_schoolbook(0, 2, 4, 5, 0, 2)
    
    # Block (i:0-1, j:4-4, k:2-3) -> 2x1x2 -> Schoolbook
    add_schoolbook(0, 2, 4, 5, 2, 4)
    
    # Block (i:0-1, j:4-4, k:4-4) -> 2x1x1 -> Schoolbook
    add_schoolbook(0, 2, 4, 5, 4, 5)

    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i*m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j*p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k*n + i][c] = val

    # Verify
    if not _validate(U, V, W, rank):
        # Fallback to schoolbook if validation fails
        rank_sb = n*m*p
        U = [[0]*rank_sb for _ in range(n*m)]
        V = [[0]*rank_sb for _ in range(m*p)]
        W = [[0]*rank_sb for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
