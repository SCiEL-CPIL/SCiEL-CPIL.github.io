import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Compare with the standard matrix multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_structured_patterns(n, m, p, deadline):
    """Try various structured patterns that might give lower rank."""
    best_rank = n * m * p
    best_decomp = None
    
    # Pattern 1: Standard schoolbook
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
    best_rank = rank
    best_decomp = (U, V, W)
    
    # Pattern 2: Try to find rank via random structured search
    # Use a greedy approach: start with schoolbook, try to merge terms
    
    # Try smaller ranks with random initialization and gradient-free optimization
    import random
    random.seed(42)
    
    # Try ranks from best_rank down to a minimum
    min_rank = max(1, best_rank - 10)
    
    for target_rank in range(best_rank - 1, min_rank - 1, -1):
        if time.time() > deadline - 5:
            break
        
        # Try several random initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Scale to make values reasonable
            U /= np.sqrt(target_rank)
            V /= np.sqrt(target_rank)
            W /= np.sqrt(target_rank)
            
            # Check if this works (unlikely for random, but try structured patterns)
            # Instead, try to construct from known low-rank patterns
            
            # Try a pattern where we share computation
            # For 4x5 * 5x5, try rank = 4*5 + 5*5 - something
            pass
    
    # Try a specific structured pattern: block decomposition
    # Split the 4x5 matrix into blocks and the 5x5 into blocks
    
    # Pattern: Use the fact that 4 = 2*2, 5 = 5
    # Try rank = 2*2*5 + 2*5*5 + 2*5*5 + 5*5*5 = 20 + 50 + 50 + 125 = 245 (worse)
    
    # Try a different approach: assume certain symmetries
    # For now, return the best we found (schoolbook)
    
    if best_decomp is not None:
        return best_decomp
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # First, get the schoolbook decomposition
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
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find better decompositions
    result = _try_structured_patterns(n, m, p, deadline)
    if result is not None:
        U_try, V_try, W_try = result
        if U_try.shape[1] < best_rank:
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = U_try.shape[1]
    
    # If time remains, try a more systematic search
    if time.time() < deadline - 20:
        # Try to find rank by examining the tensor structure
        # The matrix multiplication tensor T[i,j,k] = delta_ik (for i,k indices)
        # We need to find the tensor rank
        
        # Use a greedy merging approach
        # Start with schoolbook, try to merge pairs of terms
        
        current_U = best_U.copy()
        current_V = best_V.copy()
        current_W = best_W.copy()
        current_rank = best_rank
        
        # Try to merge terms that have similar structure
        # This is a heuristic and may not work well
        
        # For now, just return what we have
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
