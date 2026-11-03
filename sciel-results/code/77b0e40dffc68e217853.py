"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    And 0 for invalid triples.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check invalid entries (sampled for speed, but exhaustive is better if fast enough)
    # Total entries: (p*n) * (n*m) * (m*p) = 12 * 10 * 30 = 3600
    # Rank is ~50, so 3600 * 50 = 180k ops. Very fast.
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                # Check consistency
                if i == i2 and j == j2 and k == k2:
                    # Valid case, already checked
                    continue
                else:
                    s = 0.0
                    for r in range(rank):
                        s += U[y, r] * V[z, r] * W[x, r]
                    if abs(s) > tol:
                        return False
    return True

def _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block and return new col."""
    # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
    U[0*m + j0, col] += 1
    U[1*m + j1, col] += 1
    V[j0*p + k0, col] += 1
    V[j1*p + k1, col] += 1
    W[k0*n + 0, col] += 1
    W[k1*n + 1, col] += 1
    col += 1
    
    # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
    U[1*m + j0, col] += 1
    U[1*m + j1, col] += 1
    V[j0*p + k0, col] += 1
    W[k0*n + 1, col] += 1
    W[k1*n + 1, col] -= 1
    col += 1
    
    # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
    U[0*m + j0, col] += 1
    V[j0*p + k1, col] += 1
    V[j1*p + k1, col] -= 1
    W[k1*n + 0, col] += 1
    W[k1*n + 1, col] += 1
    col += 1
    
    # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
    U[1*m + j1, col] += 1
    V[j1*p + k0, col] += 1
    V[j0*p + k0, col] -= 1
    W[k0*n + 0, col] += 1
    W[k0*n + 1, col] += 1
    col += 1
    
    # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
    U[0*m + j0, col] += 1
    U[0*m + j1, col] += 1
    V[j1*p + k1, col] += 1
    W[k0*n + 0, col] -= 1
    W[k1*n + 0, col] += 1
    col += 1
    
    # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
    U[1*m + j0, col] += 1
    U[0*m + j0, col] -= 1
    V[j0*p + k0, col] += 1
    V[j0*p + k1, col] += 1
    W[k1*n + 1, col] += 1
    col += 1
    
    # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
    U[0*m + j1, col] += 1
    U[1*m + j1, col] -= 1
    V[j1*p + k0, col] += 1
    V[j1*p + k1, col] += 1
    W[k0*n + 0, col] += 1
    col += 1
    
    return col

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    Split j into {0,1}, {2,3}, {4}. Split k into {0,1}, {2,3}, {4,5}.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Pairs of j indices for Strassen blocks
    j_pairs = [(0, 1), (2, 3)]
    # Pairs of k indices for Strassen blocks
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p)
            
    # Remaining j=4
    for k in range(p):
        # C_0,k = A_0,4 B_4,k
        U[0*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        
        # C_1,k = A_1,4 B_4,k
        U[1*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1
        
    assert col == rank, f"Expected col={rank}, got {col}"
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, time_limit=10):
    """
    Try to reduce the rank by finding linear dependencies among the columns of the tensors.
    This is a heuristic.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    # A proper rank reduction would involve checking if the tensor can be represented
    # with fewer terms. This is hard. 
    # Instead, we can try to merge columns if they are identical or linearly dependent
    # in a way that preserves the tensor structure.
    
    # For now, return the input as is.
    return U, V, W

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # Start with the known 54-rank decomposition
    U, V, W = _build_54_rank(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = 54
    
    # Verify the initial decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
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
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    # Return the best result found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
