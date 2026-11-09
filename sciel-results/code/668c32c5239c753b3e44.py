"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook rank n*m*p decomposition."""
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
    return U, V, W

def _check_decomposition(n, m, p, U, V, W):
    """Check if U, V, W form a valid decomposition."""
    rank = len(U[0])
    # Convert to numpy arrays for speed
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute the tensor
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T[i, j, k] to be 1 if j is the shared index, 0 otherwise.
    # Actually, T[i, j, k] = 1 for all i, j, k? No.
    # The tensor is T[I, J, K] where I = i*m+j, J = j*p+k, K = k*n+i.
    # T[I, J, K] = 1 if j_I == j_J and i_I == i_K and k_J == k_K.
    
    # Let's construct the target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                target[I, J, K] = 1.0
    
    # Compute the decomposed tensor
    # This is expensive, so we only do it for small ranks or with sampling
    # For now, we'll do a full check if rank is small enough
    if rank > 50:
        return False # Too expensive to check fully, assume invalid unless verified
    
    decomposed = np.zeros_like(target)
    for r in range(rank):
        u = U_arr[:, r]
        v = V_arr[:, r]
        w = W_arr[:, r]
        # Outer product
        decomposed += np.einsum('i,j,k->ijk', u, v, w)
    
    return np.allclose(decomposed, target, atol=1e-6)

def _try_reduce_rank(n, m, p, target_rank, max_iters=1000):
    """Try to find a decomposition of target_rank."""
    # Initialize with random values
    for _ in range(max_iters):
        U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p * n)]
        
        if _check_decomposition(n, m, p, U, V, W):
            return U, V, W
    return None

def construct():
    """Construct the rank-minimal decomposition for 3x3x4 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to reduce rank
    for rank in range(best_rank - 1, 19, -1):
        if time.time() - start_time > 120:
            break
        result = _try_reduce_rank(n, m, p, rank, max_iters=100)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = rank
            # If we find a lower rank, we can stop or continue
            # For now, we stop at the first found lower rank to save time
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
