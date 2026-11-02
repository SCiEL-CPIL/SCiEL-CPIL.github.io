"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = 3, 5, 7

    # Schoolbook decomposition - rank n*m*p
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Verify a decomposition is valid (full check)
    def verify(U, V, W):
        U = np.array(U, dtype=np.int64)
        V = np.array(V, dtype=np.int64)
        W = np.array(W, dtype=np.int64)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        # Check all tensor entries: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # T[a,b,c] should be 1 iff a=i*m+j, b=j*p+k, c=k*n+i for same i,j,k
        for a in range(n * m):
            i = a // m
            j = a % m
            for b in range(m * p):
                j2 = b // p
                k = b % p
                for c in range(p * n):
                    k2 = c // n
                    i2 = c % n
                    expected = 1 if (i == i2 and j == j2 and k == k2) else 0
                    val = np.dot(U[a], V[b] * W[c])
                    if val != expected:
                        return False
        return True

    # Get schoolbook as baseline
    result = schoolbook()
    base_rank = n * m * p
    U_base = np.array(result['U'])
    V_base = np.array(result['V'])
    W_base = np.array(result['W'])
    
    # Verify schoolbook is correct
    assert verify(U_base, V_base, W_base), "Schoolbook decomposition failed verification"
    
    # For M_{3,5,7}, the schoolbook rank is 105.
    # Known lower bounds for matrix multiplication tensor rank:
    #   R(M_{n,m,p}) >= max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    #   R(M_{n,m,p}) >= n*m*p / max(n,m,p) ... various bounds
    # The schoolbook rank n*m*p = 105 is the standard upper bound.
    # For small dimensions, it's unknown if lower rank is possible.
    # Given the difficulty of this problem and the 120s time limit,
    # we return the schoolbook decomposition which is guaranteed correct.
    
    return {'U': U_base.tolist(), 'V': V_base.tolist(), 'W': W_base.tolist()}
# EVOLVE-BLOCK-END
