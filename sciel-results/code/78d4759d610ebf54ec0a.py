"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 115.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _rational_solve(A, b):
    """Solve linear system Ax = b over rationals. A is (m,n) Fractions, b is (m,) Fractions."""
    m, n = A.shape
    M = [list(A[i]) + [b[i]] for i in range(m)]
    
    for col in range(min(m, n)):
        pivot_row = None
        for row in range(col, m):
            if M[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            return None
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]
        pivot_val = M[col][col]
        for row in range(col + 1, m):
            if M[row][col] != 0:
                factor = M[row][col] / pivot_val
                for j in range(col, m + 1):
                    M[row][j] -= factor * M[col][j]
    
    x = [Fraction(0)] * n
    for row in range(min(m, n) - 1, -1, -1):
        all_zero = True
        for j in range(row, n):
            if M[row][j] != 0:
                all_zero = False
                break
        if all_zero:
            if M[row][n] != 0:
                return None
            continue
        s = M[row][n]
        for j in range(row + 1, n):
            s -= M[row][j] * x[j]
        x[row] = s / M[row][row]
    
    for i in range(m):
        s = Fraction(0)
        for j in range(n):
            s += A[i][j] * x[j]
        if s != b[i]:
            return None
    
    return x

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition."""
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > tol:
                    return False
    
    # Check invalid entries
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                if i == i2 and j == j2 and k == k2:
                    continue
                s = np.dot(U[y, :], V[z, :] * W[x, :])
                if abs(s) > tol:
                    return False
    return True

def _build_hybrid_51_rank(n, m, p):
    """
    Hybrid decomposition: 
    - Strassen for j in {0,1}, k in {0,1,2,3} (two 2x2x2 blocks)
    - Strassen for j in {2,3}, k in {4,5} (one 2x2x2 block)
    - Standard for j=4, all k
    Total: 7 + 7 + 7 + 6*1 = 30? No wait.
    
    Actually let's be more careful. 
    j has 5 elements: {0,1}, {2,3}, {4}
    k has 6 elements: {0,1}, {2,3}, {4,5}
    
    Block (j01, k01): Strassen rank 7
    Block (j01, k23): Strassen rank 7
    Block (j23, k01): Strassen rank 7
    Block (j23, k23): Strassen rank 7
    Block (j01, k45): Standard rank 4 (2*2)
    Block (j23, k45): Standard rank 4 (2*2)
    Block (j4, k01): Standard rank 2 (1*2)
    Block (j4, k23): Standard rank 2 (1*2)
    Block (j4, k45): Standard rank 2 (1*2)
    
    Total: 7*4 + 4*2 + 2*3 = 28 + 8 + 6 = 42.
    This is better than 54! Let's implement it.
    """
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    def _add_strassen(j0, j1, k0, k1):
        nonlocal col
        # M1
        U[0*m + j0, col] += 1; U[1*m + j1, col] += 1
        V[j0*p + k0, col] += 1; V[j1*p + k1, col] += 1
        W[k0*n + 0, col] += 1; W[k1*n + 1, col] += 1
        col += 1
        # M2
        U[1*m + j0, col] += 1; U[1*m + j1, col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + 1, col] += 1; W[k1*n + 1, col] -= 1
        col += 1
        # M3
        U[0*m + j0, col] += 1
        V[j0*p + k1, col] += 1; V[j1*p + k1, col] -= 1
        W[k1*n + 0, col] += 1; W[k1*n + 1, col] += 1
        col += 1
        # M4
        U[1*m + j1, col] += 1
        V[j1*p + k0, col] += 1; V[j0*p + k0, col] -= 1
        W[k0*n + 0, col] += 1; W[k0*n + 1, col] += 1
        col += 1
        # M5
        U[0*m + j0, col] += 1; U[0*m + j1, col] += 1
        V[j1*p + k1, col] += 1
        W[k0*n + 0, col] -= 1; W[k1*n + 0, col] += 1
        col += 1
        # M6
        U[1*m + j0, col] += 1; U[0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1; V[j0*p + k1, col] += 1
        W[k1*n + 1, col] += 1
        col += 1
        # M7
        U[0*m + j1, col] += 1; U[1*m + j1, col] -= 1
        V[j1*p + k0, col] += 1; V[j1*p + k1, col] += 1
        W[k0*n + 0, col] += 1
        col += 1

    def _add_standard(j, k):
        nonlocal col
        for i in range(n):
            U[i*m + j, col] += 1
            V[j*p + k, col] += 1
            W[k*n + i, col] += 1
            col += 1

    # Strassen blocks
    _add_strassen(0, 1, 0, 1)
    _add_strassen(0, 1, 2, 3)
    _add_strassen(2, 3, 0, 1)
    _add_strassen(2, 3, 2, 3)
    
    # Standard blocks
    for k in range(2, 4): # k=4,5
        for j in [0, 1]:
            _add_standard(j, k)
    for k in range(4, 6): # k=4,5
        for j in [2, 3]:
            _add_standard(j, k)
    for k in range(6):
        _add_standard(4, k)
    
    assert col == rank, f"Expected col={rank}, got {col}"
    return U, V, W

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # Try the hybrid 42-rank decomposition first
    U, V, W = _build_hybrid_51_rank(n, m, p)
    if _verify_full(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to 54-rank Strassen
    U, V, W = _build_54_rank(n, m, p)
    if _verify_full(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Final fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # M1
            U[0*m + j0, col] += 1; U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1; V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1; W[k1*n + 1, col] += 1
            col += 1
            # M2
            U[1*m + j0, col] += 1; U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1; W[k1*n + 1, col] -= 1
            col += 1
            # M3
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1; V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1; W[k1*n + 1, col] += 1
            col += 1
            # M4
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1; W[k0*n + 1, col] += 1
            col += 1
            # M5
            U[0*m + j0, col] += 1; U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1; W[k1*n + 0, col] += 1
            col += 1
            # M6
            U[1*m + j0, col] += 1; U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            # M7
            U[0*m + j1, col] += 1; U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1; V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
    
    for k in range(p):
        U[0*m + 4, col] += 1; V[4*p + k, col] += 1; W[k*n + 0, col] += 1; col += 1
        U[1*m + 4, col] += 1; V[4*p + k, col] += 1; W[k*n + 1, col] += 1; col += 1
    
    assert col == rank, f"Expected col={rank}, got {col}"
    return U, V, W

# EVOLVE-BLOCK-END
