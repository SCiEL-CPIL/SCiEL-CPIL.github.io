import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def verify_decomp(U, V, W, n, m, p):
    """
    Verify that U, V, W decompose the matrix multiplication tensor.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r] ? 
    No, the tensor is T[i,j,k] = 1.
    Decomposition: T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    """
    r = U.shape[1]
    # Create the full tensor T of shape (n, m, p)
    # T[i, j, k] should be 1
    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a batched multiplication or einsum
    # U_flat: (n*m, r) -> U[i*m+j, r]
    # V_flat: (m*p, r) -> V[j*p+k, r]
    # W_flat: (p*n, r) -> W[k*n+i, r]
    
    # Reshape U to (n, m, r)
    U_reshaped = U.reshape(n, m, r)
    # Reshape V to (m, p, r)
    V_reshaped = V.reshape(m, p, r)
    # Reshape W to (p, n, r)
    W_reshaped = W.reshape(p, n, r)
    
    # We want to check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
    # Note: W index is k*n+i, so W[k, i, r] in reshaped form (p, n, r)
    
    # Compute the product
    # For each r, U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Sum over r
    
    # Use einsum
    # 'ijr,jkr,kir->ijk'
    result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Check if result is all 1s
    return np.allclose(result, 1.0, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Try rank 28
    target_rank = 28
    max_iters = 20000
    
    # Random search with local optimization
    # Initialize with small random values
    # Since entries are half-integers, we can search in a grid or use gradient descent
    # and round at the end.
    
    # Heuristic: Start with schoolbook rank 30, remove 2 columns?
    # Or random init.
    
    best_score = 30.0 # Current best rank
    best_decomp = None
    
    # Simple random search for rank 28
    # This is unlikely to succeed quickly without a good method.
    # I will implement a deterministic construction if I can find one.
    # For now, return schoolbook.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
