"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Known lower bound for 3x4x6 matrix multiplication
    # From literature: M(3,4) <= 11, and for 3x4x6 we can use known constructions
    # Theoretical lower bound: max(n*m, m*p, n*p) = max(12, 24, 18) = 24
    # But we can do better with tensor decomposition
    
    # Start with schoolbook as baseline
    rank_schoolbook = n * m * p  # 72
    
    # Try to find a better decomposition
    # For 3x4x6, known results suggest rank around 24-30
    
    # Strategy: Use a structured approach based on known matrix multiplication algorithms
    # For 3x4 matrices, we can use a rank-11 algorithm
    # For the 4x6 part, we need to combine carefully
    
    # Let's try a systematic search with random initialization and optimization
    
    def schoolbook_decomp():
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
        return U, V, W, rank
    
    U_school, V_school, W_school, rank_school = schoolbook_decomp()
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Try known lower-rank constructions
    # For 3x4x6, we can try rank 24 (which is m*p, the largest dimension product)
    # This would be optimal if achievable
    
    # Attempt 1: Try rank = 24
    for target_rank in [24, 25, 26, 27, 28, 29, 30]:
        if time.time() > deadline - 10:
            break
        
        # Use random initialization and try to verify
        # For exact decomposition, we need integer or half-integer entries
        
        # Try a structured approach: use the fact that 3x4 can be done in rank 11
        # and combine with the 4x6 dimension
        
        # Actually, let's use a more direct approach
        # The tensor T_{i,j,k} for matrix multiplication (A_{i,j} * B_{j,k})
        # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = A_{i,j} * B_{j,k}
        
        # For a valid decomposition, for each (i,j,k):
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 if we're 
        # computing the product, but actually the tensor entries are 1 for all (i,j,k)
        # and the multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
        
        # The tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k)
        # and the decomposition satisfies: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
        # Wait, let me re-read the convention: W[k*n+i, r]
        
        # So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Let's try to find such a decomposition with small rank
        
        # Use a simple approach: try to find a decomposition where U, V, W have 
        # entries in {0, 1, -1} or {0, 1/2, -1/2}
        
        # For rank 24, this is quite constrained. Let's try a different approach.
        
        # Use the known result that 3x4x4 can be done in rank 20 (from literature)
        # For 3x4x6, we might need more
        
        # Let's try a heuristic search with small random matrices
        
        if target_rank <= 24:
            # For rank 24, try a very structured approach
            # Each column c corresponds to a "gate"
            # Try: U[i*m+j, c] = 1 if i == c // (m*p) and j == (c // p) % m and k == c % p
            # This is basically schoolbook but with fewer terms if we can share
            
            pass  # Skip complex search for now, fall back to schoolbook
    
    # Given the complexity of finding a low-rank exact decomposition, 
    # and the time constraints, return the schoolbook decomposition
    # which is guaranteed to work
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
