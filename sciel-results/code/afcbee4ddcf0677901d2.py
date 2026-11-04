"""Matrix multiplication 2x5x6 - rank-minimal decomposition via block partitioning."""
import numpy as np
from functools import lru_cache

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Rank-7 Strassen decomposition for 2x2x2."""
    U = np.array([
        [1,  0,  0,  0,  0,  0,  0],  # i=0, m=0
        [0,  1,  0,  0,  0,  0,  0],  # i=0, m=1
        [0,  0,  1,  0,  1,  0,  0],  # i=1, m=0
        [0,  0,  1,  0,  0,  1,  0],  # i=1, m=1
    ], dtype=int)
    V = np.array([
        [1,  1,  0,  0,  0,  0,  0],  # m=0, k=0
        [0,  0,  1,  1,  0,  0,  0],  # m=0, k=1
        [1, -1,  0,  0,  0,  0,  0],  # m=1, k=0
        [0,  0,  1, -1,  0,  0,  0],  # m=1, k=1
    ], dtype=int)
    # W[k*n+i, r] - k=0..1, n=2, i=0..1
    W = np.array([
        [0,  0,  1,  0,  0,  0,  0],  # k=0, i=0
        [1,  1,  0,  0,  0,  0,  0],  # k=0, i=1
        [0,  0,  0,  0,  1,  1,  1],  # k=1, i=0
        [0,  0,  0,  0,  0,  1,  0],  # k=1, i=1
    ], dtype=int)
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition: rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _combine_along_m(n, m1, m2, p, U1, V1, W1, U2, V2, W2):
    """Combine two decompositions split along middle dimension m = m1 + m2."""
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 + rank2
    
    U = np.zeros((n * (m1 + m2), total_rank), dtype=int)
    V = np.zeros(((m1 + m2) * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Left part: j in 0..m1-1
    for i in range(n):
        for j in range(m1):
            U[i * (m1 + m2) + j, :rank1] = U1[i * m1 + j, :]
    for j in range(m1):
        for k in range(p):
            V[j * p + k, :rank1] = V1[j * p + k, :]
    for k in range(p):
        for i in range(n):
            W[k * n + i, :rank1] = W1[k * n + i, :]
    
    # Right part: j in 0..m2-1 (offset by m1)
    for i in range(n):
        for j in range(m2):
            U[i * (m1 + m2) + (m1 + j), rank1:] = U2[i * m2 + j, :]
    for j in range(m2):
        for k in range(p):
            V[(m1 + j) * p + k, rank1:] = V2[j * p + k, :]
    for k in range(p):
        for i in range(n):
            W[k * n + i, rank1:] = W2[k * n + i, :]
    
    return U, V, W


def _combine_along_p(n, m, p1, p2, U1, V1, W1, U2, V2, W2):
    """Combine two decompositions split along output dimension p = p1 + p2."""
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 + rank2
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * (p1 + p2), total_rank), dtype=int)
    W = np.zeros(((p1 + p2) * n, total_rank), dtype=int)
    
    # Left part: k in 0..p1-1
    for i in range(n):
        for j in range(m):
            U[i * m + j, :rank1] = U1[i * m + j, :]
    for j in range(m):
        for k in range(p1):
            V[j * (p1 + p2) + k, :rank1] = V1[j * p1 + k, :]
    for k in range(p1):
        for i in range(n):
            W[k * n + i, :rank1] = W1[k * n + i, :]
    
    # Right part: k in 0..p2-1 (offset by p1)
    for i in range(n):
        for j in range(m):
            U[i * m + j, rank1:] = U2[i * m + j, :]
    for j in range(m):
        for k in range(p2):
            V[j * (p1 + p2) + (p1 + k), rank1:] = V2[j * p2 + k, :]
    for k in range(p2):
        for i in range(n):
            W[(p1 + k) * n + i, rank1:] = W2[k * n + i, :]
    
    return U, V, W


def _combine_along_n(n1, n2, m, p, U1, V1, W1, U2, V2, W2):
    """Combine two decompositions split along input dimension n = n1 + n2."""
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 + rank2
    
    U = np.zeros(((n1 + n2) * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * (n1 + n2), total_rank), dtype=int)
    
    # Top part: i in 0..n1-1
    for i in range(n1):
        for j in range(m):
            U[i * m + j, :rank1] = U1[i * m + j, :]
    for j in range(m):
        for k in range(p):
            V[j * p + k, :rank1] = V1[j * p + k, :]
    for k in range(p):
        for i in range(n1):
            W[k * (n1 + n2) + i, :rank1] = W1[k * n1 + i, :]
    
    # Bottom part: i in 0..n2-1 (offset by n1)
    for i in range(n2):
        for j in range(m):
            U[(n1 + i) * m + j, rank1:] = U2[i * m + j, :]
    for j in range(m):
        for k in range(p):
            V[j * p + k, rank1:] = V2[j * p + k, :]
    for k in range(p):
        for i in range(n2):
            W[k * (n1 + n2) + (n1 + i), rank1:] = W2[k * n2 + i, :]
    
    return U, V, W


@lru_cache(maxsize=None)
def _decompose(n, m, p):
    """
    Find minimal-rank decomposition for n x m x p matrix multiplication.
    Returns (U, V, W) as numpy arrays.
    """
    # Base cases
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
    
    # Very small cases: use schoolbook
    if n * m * p <= 8:
        return _schoolbook(n, m, p)
    
    # Try all possible splits along each dimension
    best_rank = n * m * p
    best_result = None
    
    # Split along m: m = m1 + m2
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = _decompose(n, m1, p)
        U2, V2, W2 = _decompose(n, m2, p)
        total_rank = U1.shape[1] + U2.shape[1]
        if total_rank < best_rank:
            best_rank = total_rank
            U, V, W = _combine_along_m(n, m1, m2, p, U1, V1, W1, U2, V2, W2)
            best_result = (U, V, W)
    
    # Split along p: p = p1 + p2
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = _decompose(n, m, p1)
        U2, V2, W2 = _decompose(n, m, p2)
        total_rank = U1.shape[1] + U2.shape[1]
        if total_rank < best_rank:
            best_rank = total_rank
            U, V, W = _combine_along_p(n, m, p1, p2, U1, V1, W1, U2, V2, W2)
            best_result = (U, V, W)
    
    # Split along n: n = n1 + n2
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = _decompose(n1, m, p)
        U2, V2, W2 = _decompose(n2, m, p)
        total_rank = U1.shape[1] + U2.shape[1]
        if total_rank < best_rank:
            best_rank = total_rank
            U, V, W = _combine_along_n(n1, n2, m, p, U1, V1, W1, U2, V2, W2)
            best_result = (U, V, W)
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _decompose(n, m, p)
    
    # Convert to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
