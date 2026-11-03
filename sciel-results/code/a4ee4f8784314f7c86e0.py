"""Matrix multiplication 2x3x6 tensor decomposition; structured search with verification."""
DIMENSIONS = [2, 3, 6]

import time
import numpy as np

_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 150s wall-clock kill

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Matmul tensor T[i,j,k]=1 for all valid (i,j,k)."""
    return np.ones((n, m, p))


def _verify(U, V, W, n, m, p):
    """Check exact decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]==1."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
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


def _matmul_as_matrix(n, m, p):
    """The matmul tensor as a matrix of shape (n*m, p*n) via:
    M[i*m+j, k*n+i] = delta_{j,?} ... Actually the matmul tensor contracted:
    We want a matrix such that rank of tensor = rank of this "slice".
    For fixed j, the tensor is U_i * V_k (outer). The full tensor rank equals
    the rank of the matrix obtained by flattening: M[row=(i,j), col=(k)] with
    M[i*m+j, k*n+i] is not a simple matrix. Instead use the standard result:
    rank(T) = min rank r s.t. T = sum_r u_r v_r w_r.
    This equals the rank of the matrix M of size (n*m) x (p*n) where
    M[(i,j),(k,?)] ... Let's use: reshape T to (n*m, p) then the tensor rank
    is at least the rank of any matricization. We'll use the (n*m) x (p*n)
    matricization: M[i*m+j, k*n+i] = T[i,j,k].
    """
    M = np.zeros((n * m, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i * m + j, k * n + i] = 1.0
    return M


def _structured_search(n, m, p, max_rank, time_budget):
    """Try to find a rank-r decomposition below schoolbook using iterative methods."""
    T = _build_tensor(n, m, p)
    best = None
    for r in range(max_rank, 1, -1):
        if time.time() - _START > time_budget:
            break
        found = _search_rank_r(n, m, p, r, T)
        if found is not None:
            best = found
            break
    return best


def _search_rank_r(n, m, p, r, T, max_iters=2000):
    """Iterative coordinate descent to find rank-r decomposition."""
    np.random.seed(12345)
    U = np.random.randn(n * m, r) * 0.3
    V = np.random.randn(m * p, r) * 0.3
    W = np.random.randn(p * n, r) * 0.3
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    lr = 0.05
    for it in range(max_iters):
        if time.time() - _START > 110.0:
            break
        # Compute loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                    loss += (s - 1.0) ** 2
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if best_loss < 1e-12:
            if _verify(best_U, best_V, best_W, n, m, p):
                return best_U, best_V, best_W
        # Update U
        for idx in range(n * m):
            i, j = idx // m, idx % m
            for c in range(r):
                grad = 0.0
                for k in range(p):
                    s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                    grad += (s - 1.0) * V[j * p + k, c] * W[k * n + i, c]
                U[idx, c] -= lr * grad
        # Update V
        for idx in range(m * p):
            j, k = idx // p, idx % p
            for c in range(r):
                grad = 0.0
                for i in range(n):
                    s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                    grad += (s - 1.0) * U[i * m + j, c] * W[k * n + i, c]
                V[idx, c] -= lr * grad
        # Update W
        for idx in range(p * n):
            k, i = idx // n, idx % n
            for c in range(r):
                grad = 0.0
                for j in range(m):
                    s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                    grad += (s - 1.0) * U[i * m + j, c] * V[j * p + k, c]
                W[idx, c] -= lr * grad
    if best_loss < 1e-10:
        if _verify(best_U, best_V, best_W, n, m, p):
            return best_U, best_V, best_W
    return None


def construct():
    deadline = _START + _WALL_LIMIT
    n, m, p = DIMENSIONS

    # First, verify the matricization rank gives a lower bound.
    # The tensor rank is at least the rank of any matricization.
    M = _matmul_as_matrix(n, m, p)
    mat_rank = np.linalg.matrix_rank(M, tol=1e-10)
    # mat_rank for 2x3x6: M is (6) x (12). Let's compute.
    # M[i*m+j, k*n+i] = 1. For n=2,m=3,p=6: rows=6, cols=12.
    # Row (i,j): has 1s at columns k*n+i for all k. So row (i,j) has 1s at
    # positions {i, i+n, i+2n, ...} = {i + k*2 for k=0..5} = {i, i+2, i+4, i+6, i+8, i+10}
    # For i=0: cols 0,2,4,6,8,10. For i=1: cols 1,3,5,7,9,11.
    # Rows with same i are identical. So rows (0,0),(0,1),(0,2) are identical,
    # rows (1,0),(1,1),(1,2) are identical. Rank = 2.
    # So matricization rank is 2, which is a weak lower bound.
    # The true tensor rank is at least max(n*p, m*p, n*m, ...) but more precisely
    # for matmul the lower bound is max(n*p, m*p, n*m, n*m*p / something).
    # Actually the standard lower bound is max(n*p, m*p, n*m) but that's not tight.
    # For 2x3x6, schoolbook = 36. We'll search for rank < 36.

    # Try structured search for rank < 36.
    schoolbook_rank = n * m * p  # 36
    found = None
    # Search from a reasonable upper bound downward
    # Given the small size, try ranks from 35 down, but limit search time.
    search_budget = 100.0
    found = _structured_search(n, m, p, schoolbook_rank - 1, search_budget)

    if found is not None:
        U, V, W = found
    else:
        U, V, W = _schoolbook(n, m, p)

    # Convert to list-of-lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    # Final verification
    if not _verify(U_list, V_list, W_list, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
