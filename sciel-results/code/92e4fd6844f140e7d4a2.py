"""Tensor rank minimization for 4x4 * 4x9 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return schoolbook factor matrices (rank = n*m*p)."""
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


def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the MM tensor."""
    # Build the tensor T[i,j,k] = 1 iff (i,j,k) is a valid MM term
    # The MM tensor is: T = sum over (i,j,k) of e_i ⊗ e_j ⊗ e_k
    # where the output C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has 1 at (i,j,k) for all valid (i,j,k)
    # In the factorization: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all (i,j,k)
    
    # Compute the reconstructed tensor
    # For each (i,j,k), compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is a 3D array of shape (n, m, p)
    
    # Efficient computation: 
    # For fixed j: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # = sum_i sum_k (sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c])
    
    # Let's just compute it directly for small sizes
    n, m, p = 4, 4, 9
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True


def _try_rank_reduction(U, V, W, n, m, p, max_rank_reduction=2):
    """Try to reduce rank by zeroing out some factors and re-checking."""
    # This is a heuristic: try setting some entries to 0 and see if decomposition still holds
    # For a proper approach we'd use optimization, but let's try a simple version
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try removing one term at a time (greedy)
    for attempt in range(max_rank_reduction):
        # Try to merge or remove terms
        # For now, just try setting some entries to 0
        found = False
        for c in range(U.shape[1]):
            # Try zeroing out column c in all three matrices
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, c] = 0
            V_test[:, c] = 0
            W_test[:, c] = 0
            if _verify(U_test, V_test, W_test, n, m, p):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = best_rank - 1
                found = True
                break
        if not found:
            break
    
    return best_U, best_V, best_W, best_rank


def construct():
    start_time = time.time()
    deadline = 145  # seconds, with 5s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    rank = n * m * p
    
    # Try to reduce rank
    U, V, W, rank = _try_rank_reduction(U, V, W, n, m, p, max_rank_reduction=1)
    
    # Convert to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
