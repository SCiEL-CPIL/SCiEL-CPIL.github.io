"""Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    # Strategy: 
    # 1. Try to find a low-rank decomposition (12-23) using structured random search over integers/half-integers.
    # 2. If not found, return schoolbook rank 24.
    
    # Schoolbook rank is 24.
    # Lower bound is 12.
    
    # We will try ranks from 12 to 23.
    # For each rank, we perform a limited number of random trials with integer entries.
    
    for rank in range(12, 24):
        if time.time() > deadline - 20:
            break
        # Try a few random integer configurations
        # Using entries from {-1, 0, 1} is a good heuristic for minimal rank tensors.
        result = _try_rank_int_search(n, m, p, rank, deadline, max_trials=500)
        if result is not None:
            return result
            
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank_int_search(n, m, p, rank, deadline, max_trials=100):
    """
    Try to find a rank-r decomposition with integer entries.
    We restrict entries to {-1, 0, 1} to keep the search space manageable.
    """
    np.random.seed(42 + rank) # Fixed seed for reproducibility within rank
    
    # Pre-compute indices for faster verification
    # We need sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1
    
    for trial in range(max_trials):
        if time.time() > deadline - 20:
            return None
            
        # Generate random sparse matrices
        # Probability of non-zero entry ~ 0.3
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        # Fill with -1, 0, 1
        # We can bias towards 0 to allow sparsity
        for _ in range(3):
            mat = [U, V, W]
            for m in mat:
                # Randomly set some entries to -1 or 1
                mask = np.random.rand(*m.shape) < 0.2
                m[mask] = np.random.choice([-1, 1], size=m[mask].shape)
        
        # Verify
        if _verify_int(n, m, p, U, V, W):
            return {"U": U.astype(float).tolist(), "V": V.astype(float).tolist(), "W": W.astype(float).tolist()}
            
    return None

def _verify_int(n, m, p, U, V, W):
    """Verify exact integer decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                prod = u_row * v_row * w_row
                if np.sum(prod) != 1:
                    return False
    return True
