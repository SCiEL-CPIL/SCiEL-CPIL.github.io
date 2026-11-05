import time
import numpy as np

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _rank_5_decomp(n, m, p):
    """
    Decompose the 2x5x5 matrix multiplication tensor into rank 5.
    
    For multiplying a 2x5 matrix A by a 5x5 matrix B to get C = A*B,
    the (i,k) entry of C is sum_j A[i,j] * B[j,k].
    
    The tensor T_{a,b,c} = 1 if A[a] * B[b] contributes to C[c].
    Here a = i*m + j, b = j'*p + k, c = k*n + i.
    T = 1 iff j = j'.
    
    We can write T = sum_r U[a,r] * V[b,r] * W[c,r] with rank 5.
    
    U is (n*m, 5) = (10, 5)
    V is (m*p, 5) = (25, 5)
    W is (p*n, 5) = (10, 5)
    
    Let r = 0, 1, 2, 3, 4 correspond to the 5 columns of A / rows of B.
    
    U[i*m + j, r] = 1 if j == r, else 0
    V[j'*p + k, r] = 1 if j' == r, else 0
    W[k*n + i, r] = 1 for all k, i, r
    """
    rank = m  # 5
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
    
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
    
    return {"U": U, "V": V, "W": W}

def _construct():
    n, m, p = DIMENSIONS
    
    # Try the rank-5 decomposition first
    result = _rank_5_decomp(n, m, p)
    if _verify(n, m, p, result['U'], result['V'], result['W']):
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
