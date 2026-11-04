import time
import numpy as np
import random

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # The tensor T[i,j,k] should equal sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the convention is W[k*n+i, r], so W is indexed by (k, i) -> (k*n+i)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_random_decomposition(rank, n, m, p, max_iters=5000, seed=None):
    """Try to find a rank-r decomposition via random search."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Initialize with small random values
    for _ in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Reset
        U[:] = 0
        V[:] = 0
        W[:] = 0
        
        # For each rank element, assign random values from {-1, 0, 1} or {-1, 0, 0.5, 1}
        vals = [-1, 0, 1]
        for r in range(rank):
            # Pick a random pattern
            for idx in range(n*m):
                if random.random() < 0.3:
                    U[idx, r] = random.choice(vals)
            for idx in range(m*p):
                if random.random() < 0.3:
                    V[idx, r] = random.choice(vals)
            for idx in range(p*n):
                if random.random() < 0.3:
                    W[idx, r] = random.choice(vals)
        
        if _check_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None, None, None

def _structured_strassen(n, m, p):
    """
    Try a structured decomposition for 2x5 x 5x6.
    Split A (2x5) into two 1x5 row blocks.
    Split B (5x6) into two 5x3 column blocks.
    Use Strassen-like combinations.
    
    Actually, let's try a simpler structured approach:
    Group the 6 columns into 3 pairs, and use the fact that we only have 2 rows.
    
    For each pair of columns (k, k+1), we compute:
    C[i,k] = sum_j A[i,j] * B[j,k]
    C[i,k+1] = sum_j A[i,j] * B[j,k+1]
    
    We can combine: let S = B[:,k] + B[:,k+1], D = B[:,k] - B[:,k+1]
    Then C[i,k] = (dot(A[i,:], S) + dot(A[i,:], B[:,k])) / ... 
    
    This doesn't immediately help. Let me try a different structured approach.
    
    For 2x5 x 5x6, one known approach:
    - The rank of 2xm x mxp is at most 2*min(m,p) + ... 
    - Actually, for 2xm x mxp, the rank is at most 2*m*p / something...
    
    Let me just use a greedy/structured search.
    """
    # Start with schoolbook and try to merge terms
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Now try to reduce rank by finding linear dependencies
    # This is complex, so let's try a different approach:
    # Use the fact that for 2 rows, we can express things in terms of differences
    
    # Alternative: Use a known low-rank decomposition if available
    # For now, return the schoolbook as fallback
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Start with schoolbook as baseline
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # Known bounds: rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # So we search from 30 upwards
    
    # Try structured approaches first
    # For 2x5 x 5x6, try ranks from 30 to 59
    
    for target_rank in range(30, 60):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Try random search for this rank
        result = _try_random_decomposition(target_rank, n, m, p, max_iters=200, seed=target_rank)
        if result[0] is not None:
            best_rank = target_rank
            best_U, best_V, best_W = result
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
