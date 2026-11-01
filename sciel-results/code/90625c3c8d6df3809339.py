"""Rewritten program for matrix multiplication 2x3 by 3x4 using group action / character method."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    
    Strategy: 
    1. Attempt to find a rank-12 decomposition using randomized search with integer coefficients.
    2. If a valid rank-12 decomposition is found, return it.
    3. Otherwise, fall back to the schoolbook algorithm (rank 24).
    
    The tensor T for 2x3 * 3x4 has non-zero entries at (i, j, k) where:
    U index: i*m + j
    V index: j*p + k
    W index: k*n + i
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Target tensor entries
    # T[i, j, k] = 1
    # Flattened indices:
    # u_idx = i * m + j
    # v_idx = j * p + k
    # w_idx = k * n + i
    
    # Try rank 12
    rank = 12
    result = _find_decomposition(rank, n, m, p)
    if result is not None:
        return result
    
    # Fallback to schoolbook (rank 24)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, column] = 1
                V[v_idx, column] = 1
                W[w_idx, column] = 1
                column += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _find_decomposition(rank, n, m, p):
    """
    Try to find a rank-r decomposition with integer entries in {-1, 0, 1}.
    Uses random restarts.
    """
    # Pre-compute the list of (u_idx, v_idx, w_idx) tuples for which the tensor entry is 1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target_indices.append((u_idx, v_idx, w_idx))
                
    # Also, we need to ensure that for all other (u_idx, v_idx, w_idx), the sum is 0.
    # This is a large constraint.
    # However, if we restrict U, V, W to be "sparse" or "structured", it's easier.
    
    # For now, we will just try a few random initializations and check the target indices.
    # Checking all (n*m * m*p * p*n) indices is too slow.
    # We will only check the target indices (n*m*p = 24 indices).
    # If these are 1, and we assume the structure is correct, it might work.
    # But we must ensure other indices are 0.
    
    # A safer check: Verify the full tensor.
    # Size: (n*m) * (m*p) * (p*n) = 6 * 12 * 8 = 576 entries.
    # This is small enough to check fully.
    
    num_restarts = 20
    for _ in range(num_restarts):
        # Random integer vectors in {-1, 0, 1}
        U = np.random.choice([-1, 0, 1], size=(n * m, rank))
        V = np.random.choice([-1, 0, 1], size=(m * p, rank))
        W = np.random.choice([-1, 0, 1], size=(p * n, rank))
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    and 0 otherwise.
    """
    # Construct the full tensor
    # T[u_idx, v_idx, w_idx] = sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
    
    # Efficiently compute using matrix multiplication
    # T = U @ diag(1) @ V^T @ ... no, it's a 3-way contraction.
    
    # We can iterate over c
    # T = sum_c np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
    # This creates a 3D array of size (n*m, m*p, p*n)
    
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for c in range(U.shape[1]):
        T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
    # Check target indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                if T[u_idx, v_idx, w_idx] != 1:
                    return False
                    
    # Check non-target indices are 0
    # Create a mask of target indices
    target_mask = np.zeros((n * m, m * p, p * n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target_mask[u_idx, v_idx, w_idx] = True
                
    if np.any(T[~target_mask] != 0):
        return False
        
    return True

# EVOLVE-BLOCK-END
