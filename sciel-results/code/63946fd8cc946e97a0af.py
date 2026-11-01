"""Combined program: Strassen-based block decomposition for 2x2 * 2x15 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time as _time

_START = _time.time()
_DEADLINE = 115.0  # safety margin before the 150s kill

def _remaining():
    return _DEADLINE - (_time.time() - _START)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_k U[:,k] V[:,k] W[:,k] == MM tensor exactly (within tol)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    tmp = np.einsum('uk,vk->uvk', U, V)
    result = np.einsum('uvk,wk->uvw', tmp, W)
    if np.max(np.abs(result - T)) > tol:
        return False
    return True

def _strassen_222():
    """
    Return rank-7 decomposition of M_{2,2,2} tensor.
    U: shape (4, 7) - indices i*2+j
    V: shape (4, 7) - indices j*2+k
    W: shape (4, 7) - indices k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _schoolbook_221():
    """
    Return rank-4 schoolbook decomposition of M_{2,2,1} tensor.
    U: shape (4, 4) - indices i*2+j
    V: shape (2, 4) - indices j (k=0 only)
    W: shape (2, 4) - indices i (k=0 only)
    """
    n, m, p = 2, 2, 1
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_blocks(n, m, p):
    """
    Decompose 2x2 * 2x15 into 7 blocks of 2x2 (Strassen rank 7) + 1 block of 2x1 (schoolbook rank 4).
    Total rank = 7*7 + 4 = 53.
    """
    U_s, V_s, W_s = _strassen_222()
    U_sb, V_sb, W_sb = _schoolbook_221()
    
    num_strassen_blocks = p // 2  # 7
    has_odd = (p % 2 == 1)  # True
    total_rank = num_strassen_blocks * 7 + (4 if has_odd else 0)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for block_idx in range(num_strassen_blocks):
        k0 = block_idx * 2
        U[:, col:col+7] = U_s
        for j in range(m):
            for k_local in range(2):
                k_global = k0 + k_local
                V[j * p + k_global, col:col+7] = V_s[j * 2 + k_local]
        for k_local in range(2):
            k_global = k0 + k_local
            for i in range(n):
                W[k_global * n + i, col:col+7] = W_s[k_local * 2 + i]
        col += 7
    
    if has_odd:
        k0 = num_strassen_blocks * 2
        U[:, col:col+4] = U_sb
        for j in range(m):
            V[j * p + k0, col:col+4] = V_sb[j]
        for i in range(n):
            W[k0 * n + i, col:col+4] = W_sb[i]
        col += 4
    
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS

    if _remaining() < 5:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Primary: Strassen-based rank 53 construction
    U, V, W = _build_strassen_blocks(n, m, p)
    if _verify(U, V, W, n, m, p, tol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Final fallback: schoolbook (60).
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
