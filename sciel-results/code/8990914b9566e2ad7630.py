"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2():
    """Return U, V, W for 2x2 * 2x2 with Strassen rank 7."""
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, -1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, -1, 1, 1, 1, -1]
    ], dtype=float)
    return U, V, W

def _schoolbook(n, m, p):
    """Return trivial rank n*m*p decomposition for n x m * m x p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _decompose_nm_p(n, m, p):
    """Decompose n x m * m p matrix multiplication using recursive block decomposition."""
    if n == 0 or m == 0 or p == 0:
        return np.zeros((n*m, 0)), np.zeros((m*p, 0)), np.zeros((p*n, 0))
    
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
    
    if m == 1:
        rank = n * p
        U = np.zeros((n, rank))
        V = np.zeros((p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for k in range(p):
                U[i, col] = 1
                V[k, col] = 1
                W[k*n + i, col] = 1
                col += 1
        return U, V, W
    
    if n == 1:
        rank = m * p
        U = np.zeros((m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p, rank))
        col = 0
        for j in range(m):
            for k in range(p):
                U[j, col] = 1
                V[j*p + k, col] = 1
                W[k, col] = 1
                col += 1
        return U, V, W
    
    if p == 1:
        rank = n * m
        U = np.zeros((n*m, rank))
        V = np.zeros((m, rank))
        W = np.zeros((n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                U[i*m + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W
    
    # Try all splits of m into m1 + m2
    best_rank = n * m * p
    best_result = None
    
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = _decompose_nm_p(n, m1, p)
        U2, V2, W2 = _decompose_nm_p(n, m2, p)
        r1, r2 = U1.shape[1], U2.shape[1]
        total_rank = r1 + r2
        
        if total_rank < best_rank:
            best_rank = total_rank
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            # Map U1 rows: (i, j) where j in [0, m1) -> full row i*m + j
            for i in range(n):
                for j in range(m1):
                    U[i*m + j, :r1] = U1[i*m1 + j, :]
            
            # Map U2 rows: (i, j) where j in [0, m2) -> full row i*m + (m1 + j)
            for i in range(n):
                for j in range(m2):
                    U[i*m + m1 + j, r1:] = U2[i*m2 + j, :]
            
            # V1 rows: (j, k) where j in [0, m1) -> full row j*p + k
            V[:m1*p, :r1] = V1
            
            # V2 rows: (j, k) where j in [0, m2) -> full row (m1+j)*p + k
            V[m1*p:, r1:] = V2
            
            # W is the same shape and content for both
            W[:, :r1] = W1
            W[:, r1:] = W2
            
            best_result = (U, V, W)
    
    if best_result is not None:
        return best_result
    
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _decompose_nm_p(n, m, p)
    
    def to_list(mat):
        return [[int(round(x, 4)) if abs(x - round(x, 4)) < 1e-9 else float(x) for x in row] for row in mat.tolist()]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
