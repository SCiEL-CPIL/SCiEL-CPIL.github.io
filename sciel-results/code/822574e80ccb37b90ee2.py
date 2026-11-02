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

def _validate(n, m, p, U, V, W, tolerance=1e-9):
    """
    Check whether (U, V, W) exactly decomposes the matrix multiplication tensor.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for all i in [0,n), j in [0,m), k in [0,p).
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tolerance:
                    return False
    return True

def _validate_numpy(n, m, p, U, V, W, tolerance=1e-9):
    """Fast validation using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For each c, outer product structure
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u_col = U_arr[:, c].reshape(n, m)
        v_col = V_arr[:, c].reshape(m, p)
        w_col = W_arr[:, c].reshape(p, n)
        # T[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
    
    return np.allclose(T, np.ones((n, m, p)), atol=tolerance)

def _try_systematic_search(n, m, p, rank, deadline, max_attempts=200):
    """
    Systematic search using structured initialization.
    Each column c defines a rank-1 tensor component.
    We search for integer/half-integer solutions.
    """
    min_rank = max(n*m, m*p, p*n)
    if rank < min_rank:
        return None
    
    # Strategy: Use the structure that each output (i,k) depends on all j
    # Try decompositions where we group terms by output (i,k)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        random.seed(hash((n, m, p, rank, attempt)) & 0xFFFFFFFF)
        
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        # Initialize with sparse random structure
        for c in range(rank):
            # Each column: choose which indices are active
            # Use a structured pattern: for column c, pick a subset of (i,j,k)
            
            # Random but structured: pick 2-4 active indices per matrix
            n_active_u = random.randint(1, 4)
            n_active_v = random.randint(1, 4)
            n_active_w = random.randint(1, 4)
            
            u_indices = random.sample(range(n*m), min(n_active_u, n*m))
            v_indices = random.sample(range(m*p), min(n_active_v, m*p))
            w_indices = random.sample(range(p*n), min(n_active_w, p*n))
            
            for idx in u_indices:
                U[idx, c] = random.choice([-1, 1]) * random.choice([0.5, 1.0])
            for idx in v_indices:
                V[idx, c] = random.choice([-1, 1]) * random.choice([0.5, 1.0])
            for idx in w_indices:
                W[idx, c] = random.choice([-1, 1]) * random.choice([0.5, 1.0])
        
        # Validate
        if _validate_numpy(n, m, p, U.tolist(), V.tolist(), W.tolist(), tolerance=1e-8):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_block_decomposition(n, m, p):
    """
    Try a block-structured decomposition.
    For 3x4x5, we can try to exploit the fact that 4 = 2+2 and 5 = 2+3.
    """
    # This is a heuristic - try known patterns
    # Schoolbook is rank 60. Can we do better?
    
    # One approach: use the identity that for fixed i, 
    # sum_j A[i,j] * B[j,k] can be computed with m multiplications
    # Total: n * m * p = 60
    
    # Try to find a decomposition with rank < 60
    # The Strassen algorithm for 2x2x2 gives rank 7 vs 8
    # For 3x4x5, we might be able to use block Strassen
    
    # For now, return None and let the search handle it
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, leave 10s margin
    
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 60
    
    min_possible_rank = max(n * m, m * p, p * n)  # 20
    
    # Try structured decomposition first
    result = _try_block_decomposition(n, m, p)
    if result is not None:
        best = result
        best_rank = len(result['U'][0])
    
    # Systematic search for lower ranks
    for rank in range(best_rank - 1, min_possible_rank - 1, -1):
        if time.time() > deadline - 5:
            break
        
        remaining = deadline - time.time()
        if remaining < 1:
            break
        
        # Scale attempts based on remaining time
        max_attempts = min(100, max(10, int(remaining / 0.5)))
        
        result = _try_systematic_search(n, m, p, rank, deadline, max_attempts)
        if result is not None:
            best = result
            best_rank = rank
            # Found a valid decomposition, continue to lower ranks
            continue
    
    return best
# EVOLVE-BLOCK-END
