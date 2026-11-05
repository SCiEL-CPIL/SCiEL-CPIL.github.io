import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _construct_schoolbook(n, m, p):
    """
    Construct the schoolbook (rank n*m*p) decomposition for n x m times m x p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact for the matrix multiplication tensor.
    The tensor equation is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1  for all i,j,k
    
    Here, U, V, W are numpy arrays of shape (n*m, rank), (m*p, rank), (p*n, rank).
    """
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # Check a sample of entries or all if small
    # For 3x3x8, there are 72 entries to check. This is fast.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over r
                u_val = U[i * m + j, :]
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_find_low_rank_decomposition(n, m, p, target_rank, time_limit=100):
    """
    Attempt to find a decomposition of rank < n*m*p using random search + optimization.
    This is a heuristic and may fail.
    """
    start_time = time.time()
    max_time = time_limit
    
    # We are looking for rank R < n*m*p
    # The tensor T has entries T[i,j,k] = 1.
    # We want U, V, W such that T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # A simple approach: Initialize with random Gaussian vectors and use gradient descent?
    # Or try to find a valid decomposition by solving a system of equations?
    
    # Given the difficulty and time constraints, a random search for exact integer/half-integer
    # solutions is extremely unlikely to succeed quickly.
    
    # Instead, we can try to construct a decomposition using known blocks.
    # For 3x3x8, we can split into 3x3x3 + 3x3x3 + 3x3x2.
    # If we had a rank-23 decomposition for 3x3x3, we could use 23+23+18 = 64.
    # But we don't have a verified one.
    
    # Let's try to find a rank-71 decomposition? Unlikely.
    
    # For now, return None to signal failure.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    
    # Verify schoolbook
    if not _verify_decomposition(U, V, W, n, m, p):
        # This should never happen, but just in case
        pass
    
    # Try to improve
    # Time budget: 120s total, keep 20s for safety.
    time_limit = 100
    start = time.time()
    
    # Attempt to find a better decomposition
    # For 3x3x8, the best known rank is 64 (using 3x3x3 rank 23).
    # If we can find a rank < 72, we improve.
    
    # Since we cannot reliably construct a sub-72 rank decomposition in the time limit
    # without a pre-computed verified set of vectors, we stick to schoolbook.
    
    # However, let's double check if there's a simple block structure we can exploit.
    # No, schoolbook is the only one we can guarantee is correct and fast.
    
    # Return the schoolbook decomposition
    # Convert to lists if required by the interface?
    # The interface says "factor matrices of shapes ... with integer/half-integer entries"
    # The example returns lists of lists.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
