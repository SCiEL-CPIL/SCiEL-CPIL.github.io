"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _build_rank42():
    """Schoolbook decomposition of rank n*m*p = 42."""
    n, m, p = 2, 3, 7
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
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the 2x3x7 matrix multiplication tensor."""
    n, m, p = 2, 3, 7
    # Tensor T[i,j,k] = delta_{i,k} * delta_{j,j} ... 
    # Actually the tensor for (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The multiplication tensor has entries T[i, j, k] = 1 if the index corresponds to (i,j) in U, (j,k) in V, (k,i) in W
    # More precisely: T[(i*m+j), (j*p+k), (k*n+i)] = 1
    # We check: sum_r U[a,r] * V[b,r] * W[c,r] == T[a,b,c] for all a,b,c
    
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Compute product
    # For each column r, add outer product of U[:,r], V[:,r], W[:,r]
    result = np.zeros((n * m, m * p, p * n))
    for r in range(U.shape[1]):
        result += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    return np.allclose(result, T, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with schoolbook rank 42
    U_best, V_best, W_best = _build_rank42()
    
    # Try to improve using random search / iterative refinement
    # The idea: try rank r < 42 and see if we can find a valid decomposition
    # Use the fact that the tensor is sparse and structured
    
    # Strategy: Try ranks from 39 down to 35
    # For each rank, do random initialization and check
    # Given the 130s budget, we can do a limited search
    
    for target_rank in [39, 38, 37, 36, 35]:
        if time.time() > deadline - 30:
            break
        
        # Try several random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            
            # Random initialization with small values
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            # Check if this is close to valid (probably not, but we can try a few gradient steps)
            # This is very unlikely to work with pure random init
            # Instead, let's try a structured approach
            
            # Structured: modify the rank-42 solution by zeroing out some columns
            # and adjusting others
            pass
        
        # Try structured search: take rank-42 and try to merge/eliminate some terms
        # This is complex; for now, just return rank 42 if we can't find better quickly
    
    # Return the best solution found (rank 42 schoolbook)
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
