"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build expected tensor Texp
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    
    # Compute actual tensor T = sum_r U[:,r] V[:,r] W[:,r]
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This can be computed efficiently by iterating over r
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    err = np.max(np.abs(T - Texp))
    return err < tol

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _construct_block_decomp(n, m, p):
    """
    Construct a decomposition using block structure.
    For 2x4 x 4x7, we can split the 4 dimension into two 2s.
    A = [A1 A2] (2x2 each), B = [B1; B2] (2x7 each).
    C = A1 B1 + A2 B2.
    
    Each term A1 B1 and A2 B2 is a 2x2 x 2x7 multiplication.
    The rank of 2x2 x 2xp is p (as derived in inspiration 2).
    So total rank is 2 * 7 = 14.
    
    Let's construct this explicitly.
    """
    n, m, p = 2, 4, 7
    
    rank = 2 * p  # 14
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 14
    V = [[0] * rank for _ in range(m * p)]  # 28 x 14
    W = [[0] * rank for _ in range(p * n)]  # 14 x 14
    
    # Part 1: A1 B1
    # A1 is A[0:2, 0:2], B1 is B[0:2, 0:7]
    # U indices for A1: i in 0..1, j in 0..1 -> i*m+j = i*4+j
    # V indices for B1: j in 0..1, k in 0..6 -> j*p+k = j*7+k
    # W indices for C1: k in 0..6, i in 0..1 -> k*n+i = k*2+i
    
    for r in range(p):  # r = 0..6
        # U: for i in 0..1, j in 0..1
        for i in range(2):
            for j in range(2):
                U[i * 4 + j][r] = 1
        # V: for j in 0..1, k in 0..6
        for j in range(2):
            for k in range(7):
                V[j * 7 + k][r] = 1
        # W: for k in 0..6, i in 0..1, W[k*2+i, r] = 1 if k==r else 0
        for k in range(7):
            for i in range(2):
                W[k * 2 + i][r] = 1 if k == r else 0
    
    # Part 2: A2 B2
    # A2 is A[0:2, 2:4], B2 is B[2:4, 0:7]
    # U indices for A2: i in 0..1, j in 2..3 -> i*m+j = i*4+j
    # V indices for B2: j in 2..3, k in 0..6 -> j*p+k = j*7+k
    # W indices for C2: k in 0..6, i in 0..1 -> k*n+i = k*2+i
    
    for r in range(p, 2 * p):  # r = 7..13
        r_local = r - p  # 0..6
        # U: for i in 0..1, j in 2..3
        for i in range(2):
            for j in range(2, 4):
                U[i * 4 + j][r] = 1
        # V: for j in 2..3, k in 0..6
        for j in range(2, 4):
            for k in range(7):
                V[j * 7 + k][r] = 1
        # W: for k in 0..6, i in 0..1, W[k*2+i, r] = 1 if k==r_local else 0
        for k in range(7):
            for i in range(2):
                W[k * 2 + i][r] = 1 if k == r_local else 0
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Try the block decomposition first (rank 14)
    try:
        candidate = _construct_block_decomp(n, m, p)
        if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
            return candidate
    except Exception as e:
        pass
    
    # Fall back to schoolbook (rank 56)
    return _schoolbook(n, m, p)

# Test the construction
if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    print(f"Rank: {U.shape[1]}")
    
    # Verify
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    
    T = np.zeros((n*m, m*p, p*n))
    rank = U.shape[1]
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    err = np.max(np.abs(T - Texp))
    print(f"Max error: {err}")
    print(f"Valid: {err < 1e-9}")
