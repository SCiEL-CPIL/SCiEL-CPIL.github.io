"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard schoolbook rank n*m*p decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that (U, V, W) exactly decomposes the matrix multiplication tensor.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The tensor T[x, y, z] where x=i*m+j, y=j'*p+k, z=k'*n+i' is 1 iff j==j', i==i', k==k'.
    We check: sum_r U[x,r] * V[y,r] * W[z,r] == T[x,y,z] for all x,y,z.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Build expected tensor: T[x, y, z] = 1 if x=i*m+j, y=j*p+k, z=k*n+i for consistent i,j,k
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, else 0
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute decomposed tensor D[x,y,z] = sum_r U[x,r] * V[y,r] * W[z,r]
    # Using einsum for efficiency
    D = np.einsum('xr,yr,zr->xyz', U_arr, V_arr, W_arr)
    
    return np.allclose(D, expected, atol=tol)

def _greedy_rank_reduction(n, m, p, target_rank, max_iter=100, seed=42):
    """
    Attempt to find a decomposition of the given target rank using a greedy approach
    with rational coefficients from a small set.
    
    Strategy: Start with schoolbook (rank n*m*p) and try to merge/cancel terms
    by searching over rational coefficient perturbations.
    
    For now, this is a placeholder that tries random rational initializations
    and verifies them.
    """
    np.random.seed(seed)
    
    # Small rational coefficient set
    coeffs = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for trial in range(max_iter):
        # Random initialization with small rational coefficients
        U = np.array([[random.choice(coeffs) for _ in range(target_rank)] for _ in range(n*m)])
        V = np.array([[random.choice(coeffs) for _ in range(target_rank)] for _ in range(m*p)])
        W = np.array([[random.choice(coeffs) for _ in range(target_rank)] for _ in range(p*n)])
        
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _structured_search(n, m, p, target_rank, deadline):
    """
    Structured search for a low-rank decomposition.
    
    Uses the fact that the matmul tensor has structure: T[i*m+j, j*p+k, k*n+i] = 1.
    We try to find U, V, W with integer/half-integer entries of rank target_rank.
    
    Approach: 
    1. Start with schoolbook
    2. Try to find linear dependencies among the rank-1 terms
    3. Use null-space projection to reduce rank
    """
    # For now, this is a simplified version
    # In a real implementation, we would:
    # - Build the full tensor
    # - Compute its CP decomposition numerically
    # - Round to rational values
    # - Verify and polish
    
    # Quick check: can we do better than schoolbook?
    # For 3x4x5, the schoolbook rank is 60.
    # Known results suggest it might be possible to do better, but it's not trivial.
    
    # Try a few structured initializations
    for seed in range(10):
        if time.time() > deadline - 5:
            break
        result = _greedy_rank_reduction(n, m, p, target_rank, max_iter=5, seed=seed)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110 second limit (10s safety margin before 120s)
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook is correct
    if not _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        # This should never happen, but handle it
        pass
    
    # Try to find a lower rank decomposition
    # Start from best_rank - 1 down to some minimum
    # For 3x4x5, we'll try ranks from 59 down to 50 (10 ranks)
    # Each rank gets a few seconds
    
    time_per_rank = 8  # seconds per rank attempt
    min_rank = max(best_rank - 10, 40)  # Don't go below 40
    
    for rank in range(best_rank - 1, min_rank - 1, -1):
        if time.time() > deadline - time_per_rank:
            break
        
        # Try structured search for this rank
        result = _structured_search(n, m, p, rank, deadline)
        if result is not None:
            best = result
            best_rank = rank
            # If we found one, continue to try lower ranks
            # But be careful with time
    
    return best
# EVOLVE-BLOCK-END
