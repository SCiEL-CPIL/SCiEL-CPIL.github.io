"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
import time
import itertools
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    U_arr = U.T  # (rank, n*m)
    V_arr = V.T  # (rank, m*p)
    W_arr = W.T  # (rank, p*n)
    
    # Build the full tensor and check
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Should be 1 if (a,b,c) = (i*m+j, j*p+k, k*n+i) for some i,j,k
    # else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(np.dot(U_arr, V_arr[:, b]), W_arr[:, c])
                if abs(s - 1.0) > tol:
                    return False
    
    # Check all other entries are zero
    # Only need to check entries that could potentially be nonzero
    for a in range(n * m):
        for b in range(m * p):
            # b = j*p+k, so j = b//p, k = b%p
            j = b // p
            k = b % p
            # a = i*m+j, so i = a//m, j_a = a%m
            i_a = a // m
            j_a = a % m
            if j_a != j:
                continue
            # c = k*n+i
            for i_c in range(n):
                c = k * n + i_c
                if i_c != i_a:
                    continue
                # This is a valid (i,j,k) entry, already checked
                continue
            # For invalid entries, check they're zero
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                if j_a == j and i_c == i_a and k_c == k:
                    continue  # valid entry, already checked
                s = np.dot(np.dot(U_arr, V_arr[:, b]), W_arr[:, c])
                if abs(s) > tol:
                    return False
    return True

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_strassen_like(n, m, p):
    """Try to find a lower-rank decomposition using Strassen-like techniques.
    
    For n=2, we can try to use the fact that:
    C[0,k] = sum_j A[0,j]*B[j,k]
    C[1,k] = sum_j A[1,j]*B[j,k]
    
    By defining S_j = A[0,j]+A[1,j] and D_j = A[0,j]-A[1,j],
    we can potentially share computations.
    """
    if n != 2:
        return None
    
    # Try to construct a rank < 2*m*p decomposition
    # For now, return None as we haven't found a working construction
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with schoolbook
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Try Strassen-like approach
    if still_going():
        result = _try_strassen_like(n, m, p)
        if result is not None:
            if _verify(result['U'], result['V'], result['W'], n, m, p):
                best = result
                best_rank = result['U'][0].__len__()
    
    return best
# EVOLVE-BLOCK-END
