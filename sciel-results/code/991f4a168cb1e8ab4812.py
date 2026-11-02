"""Matrix multiplication 2x3x4 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _build_target_tensor():
    """Build the 2x3x4 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _schoolbook_decomposition():
    """Return the schoolbook (rank 24) decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _check_decomposition(U, V, W, T):
    """Check if U, V, W exactly decompose T."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Reconstruct: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Using einsum
    reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.allclose(reconstructed, T, atol=1e-10, rtol=1e-10)

def _try_random_rank_decomposition(rank, T, n, m, p, max_iterations=1000, time_limit=5.0):
    """Try to find a rank-r decomposition via random search."""
    start_time = time.time()
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Generate random vectors
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize to avoid scaling issues
        for r in range(rank):
            u_norm = np.linalg.norm(U[:, r])
            v_norm = np.linalg.norm(V[:, r])
            w_norm = np.linalg.norm(W[:, r])
            if u_norm > 1e-10 and v_norm > 1e-10 and w_norm > 1e-10:
                U[:, r] /= u_norm
                V[:, r] /= v_norm
                W[:, r] /= w_norm
        
        # Check if this decomposition works
        if _check_decomposition(U, V, W, T):
            return U, V, W
    
    return None

def _attempt_rank_search(target_rank, T, n, m, p, time_budget):
    """Attempt to find a decomposition of the target rank."""
    start_time = time.time()
    
    # Try random search with increasing time allocation
    num_trials = 5
    for trial in range(num_trials):
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2.0:
            break
        result = _try_random_rank_decomposition(
            target_rank, T, n, m, p,
            max_iterations=200,
            time_limit=max(0.5, remaining / (num_trials - trial))
        )
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    # Build target tensor
    T = _build_target_tensor()
    
    # Try rank 23
    result = _attempt_rank_search(23, T, n, m, p, 30)
    if result is not None and time.time() < deadline:
        U, V, W = result
        if _check_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook rank 24
    U, V, W = _schoolbook_decomposition()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
