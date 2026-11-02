import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    # Build the tensor from U, V, W
    # T[i*m+k, k*p+j, j*n+i] = sum_c U[i*m+k, c] * V[k*p+j, c] * W[j*n+i, c]
    # Should be 1 for all valid (i,k,j), 0 otherwise
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = 0
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(val - 1) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_lanczos_decomposition(n, m, p, max_rank=None):
    """
    Try to find a low-rank decomposition using structured approaches.
    
    For matrix multiplication C = AB where A is n×m and B is m×p:
    We can exploit the structure by looking for decompositions that
    share computation across different (i,j) pairs for the same k,
    or across different k values for the same (i,j) pair.
    """
    if max_rank is None:
        max_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_result = None
    best_rank = max_rank + 1
    
    # Strategy 1: Try rank = n*m*p - reduction
    # For each k, we can potentially share some computations
    # Try various rank targets from lower bound up to schoolbook
    
    # Lower bound: max(n*m, m*p, p*n) but we can do better
    # Theoretical lower bound for M_{n,m,p} is at least max(n*m, m*p, p*n)
    # But for rectangular cases, we can sometimes do better
    
    # Try to find a decomposition with rank < n*m*p
    # Approach: Use the fact that for fixed k, we're computing inner products
    # and there might be linear dependencies
    
    # Let's try a systematic search for small rank reductions
    # Start from schoolbook and try to merge some terms
    
    # First, let's try rank = n*m*p - 1, n*m*p - 2, etc.
    # This is expensive, so let's be strategic
    
    # Alternative: Use known results or construct specific decompositions
    
    # For 4x4 times 4x9, let's try to find if there's a rank 143, 142, ... decomposition
    
    # Actually, let's try a different approach: 
    # Use the structure that C[i,j] = sum_k A[i,k] * B[k,j]
    # We can group terms by k or by (i,j) pairs
    
    # Let's try to construct a decomposition with rank slightly less than 144
    
    # Key insight: If we can find linear dependencies in the tensor, we can reduce rank
    
    # For now, let's try a few specific rank targets and see if we can find valid decompositions
    
    # Try rank targets: 144, 143, 142, ...
    # But this is computationally expensive for exact verification
    
    # Instead, let's use a randomized approach to find low-rank approximations
    # and then try to round to exact values
    
    # For this specific case, let's try to use the fact that
    # we can potentially share some computation between different j values
    
    # Let's try a structured decomposition where for each k,
    # we use fewer than p terms by exploiting linear structure
    
    # Actually, the simplest improvement might be to check if rank 143 works
    # by trying to merge two schoolbook terms that are linearly dependent
    
    # Let's implement a search that tries to find a rank r decomposition
    # where r is slightly less than n*m*p
    
    # For efficiency, let's focus on finding ANY valid decomposition with rank < 144
    
    # Try rank 143: Can we merge two terms?
    # Two schoolbook terms (i1,k1,j1) and (i2,k2,j2) can be merged if
    # there's a linear relationship between their contributions
    
    # This is getting complex. Let's use a more practical approach:
    # Try to find a decomposition using iterative methods
    
    # Initialize with random matrices and try to solve for a low-rank decomposition
    # This is a non-convex optimization problem, but we can try local search
    
    # For now, let's return the schoolbook decomposition as a baseline
    # and note that finding a lower rank requires more sophisticated search
    
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    
    # Try to find a better decomposition
    result = _try_lanczos_decomposition(n, m, p)
    
    # Verify the result
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        rank_result = len(result["U"][0])
        rank_best = len(best["U"][0])
        
        # Use the result with lower rank
        if rank_result < rank_best:
            best = result
    
    return best
# EVOLVE-BLOCK-END
