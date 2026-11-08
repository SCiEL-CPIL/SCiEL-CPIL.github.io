import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START

def _build_rank17():
    """Build the known rank-17 decomposition."""
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return U, V, W

def _verify(U, V, W):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    # Build the multiplication tensor T[i,j,k] = delta for C[k,i] from A[i,j]*B[j,k]
    # Using trace convention: W[k*n+i, r]
    # The tensor is: for each (i,j,k), the coefficient of A[i,j]*B[j,k] in C[k,i] is 1
    # So T[(i,j),(j,k),(k,i)] = 1 in the right indexing
    
    # Compute the decomposed tensor: sum_r U[a,r]*V[b,r]*W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    decomposed = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    return np.allclose(decomposed, target, atol=1e-10)

def _try_remove_term(U, V, W, r):
    """Try to see if term r can be removed by adjusting others. Returns modified arrays or None."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Remove term r, try to solve for corrections
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    U_new[:, r] = 0
    V_new[:, r] = 0
    W_new[:, r] = 0
    
    # This is a heuristic - just check if zeroing gives valid result (unlikely)
    if _verify(U_new, V_new, W_new):
        return U_new, V_new, W_new
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with rank 17
    U, V, W = _build_rank17()
    
    # Verify our baseline
    if not _verify(U, V, W):
        # Should not happen, but safety check
        pass
    
    # Try to find rank 16 by attempting to merge/remove terms
    # Strategy: try removing each term and see if we can adjust others
    # This is a heuristic search - try small perturbations
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # Try a simple approach: for each pair of terms, check if one can be absorbed
        # by modifying the other (linear dependency check)
        rank = U.shape[1]
        for r in range(rank):
            if time.time() > deadline - 5:
                break
            result = _try_remove_term(U, V, W, r)
            if result is not None:
                U, V, W = result
                break
    
    # Final verification
    if not _verify(U, V, W):
        # Fallback to known good decomposition
        U, V, W = _build_rank17()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
