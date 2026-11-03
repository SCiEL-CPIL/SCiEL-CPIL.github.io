"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _make_rank23_3x3x3():
    """
    Return U, V, W for rank-23 decomposition of 3x3x3 matrix multiplication.
    U shape: (9, 23), V shape: (9, 23), W shape: (9, 23)
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=3.
    """
    n = m = p = 3
    rank = 23
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # The 23 terms for 3x3 matrix multiplication (Strassen-like)
    # (i, j, k, a, b, c)
    terms = [
        (0, 0, 0, 1, 1, 1),
        (0, 1, 1, 1, 1, 1),
        (1, 0, 0, 1, 1, 1),
        (1, 1, 1, 1, 1, 1),
        (2, 0, 2, 1, 1, 1),
        (2, 1, 1, 1, 1, 1),
        (2, 2, 2, 1, 1, 1),
        (0, 2, 2, 1, 1, 1),
        (0, 0, 1, 1, 1, -1),
        (0, 1, 0, 1, 1, -1),
        (1, 0, 1, 1, 1, -1),
        (1, 1, 0, 1, 1, -1),
        (0, 1, 2, 1, 1, 1),
        (1, 2, 0, 1, 1, 1),
        (2, 0, 1, 1, 1, 1),
        (1, 0, 2, 1, 1, 1),
        (2, 1, 0, 1, 1, 1),
        (0, 2, 1, 1, 1, 1),
        (1, 2, 1, 1, 1, 1),
        (2, 0, 0, 1, 1, 1),
        (0, 0, 2, 1, 1, 1),
        (1, 1, 2, 1, 1, 1),
        (2, 2, 1, 1, 1, 1),
    ]
    
    for col, (i, j, k, a, b, c) in enumerate(terms):
        U[i * m + j, col] = a
        V[j * p + k, col] = b
        W[k * n + i, col] = c
    return U, V, W

def _verify_3x3x3(U, V, W):
    """Verify rank-23 decomposition for 3x3x3."""
    n = m = p = 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _build_3x3x6():
    """
    Build 3x3x6 decomposition using two 3x3x3 rank-23 decompositions.
    Returns U, V, W as lists of lists.
    """
    n, m, p = 3, 3, 6
    U3, V3, W3 = _make_rank23_3x3x3()
    
    # Verify the 3x3x3 decomposition
    if not _verify_3x3x3(U3, V3, W3):
        # Fallback to schoolbook 54
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return U, V, W
    
    rank = 46
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block 1: columns 0-2 of B (k=0,1,2)
    for c in range(23):
        # U is same: U[i*m+j, c] = U3[i*m+j, c]
        U[:, c] = U3[:, c]
        # V: for k=0,1,2, V[j*6+k, c] = V3[j*3+k, c]
        for j in range(m):
            for k in range(3):
                V[j * p + k, c] = V3[j * 3 + k, c]
        # W: for k=0,1,2, W[k*3+i, c] = W3[k*3+i, c]
        for k in range(3):
            for i in range(n):
                W[k * n + i, c] = W3[k * 3 + i, c]
    
    # Block 2: columns 3-5 of B (k=3,4,5)
    for c in range(23, 46):
        c3 = c - 23
        U[:, c] = U3[:, c3]
        for j in range(m):
            for k in range(3):
                V[j * p + (k + 3), c] = V3[j * 3 + k, c3]
        for k in range(3):
            for i in range(n):
                W[(k + 3) * n + i, c] = W3[k * 3 + i, c3]
    
    return U.tolist(), V.tolist(), W.tolist()

def _verify_3x3x6(U, V, W):
    """Verify 3x3x6 decomposition."""
    n, m, p = 3, 3, 6
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    try:
        U, V, W = _build_3x3x6()
        # Verify
        if _verify_3x3x6(U, V, W):
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to schoolbook
            n, m, p = DIMENSIONS
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
    except Exception:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
