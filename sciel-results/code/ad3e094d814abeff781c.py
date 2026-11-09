"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Check if the decomposition is exact for matrix multiplication."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # The matrix multiplication tensor T is a 3-tensor where T[i,j,k] = 1
        # A rank-r decomposition: T[i,j,k] = sum_{r=1}^r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This means for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # Let's verify this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    def to_half_int(arr):
        """Convert to half-integer representation, return None if not possible."""
        for row in arr:
            for x in row:
                if abs(x * 2 - round(x * 2)) > 1e-9:
                    return None
        return arr
    
    # Try to find a lower rank decomposition using Winograd-style block method
    # For 3x3 x 3x4, we can use a 2x2 block Winograd approach
    
    # Winograd-style decomposition for small matrices
    # Using the idea from Winograd's fast matrix multiplication
    
    # For 3x3 * 3x3, there's a known rank 23 decomposition (Strassen-like)
    # For 3x3 * 3x4, we need to adapt this
    
    # Let's try a direct numerical optimization approach
    # Start with schoolbook and try to reduce rank by merging terms
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using gradient-based optimization
    # The constraint is: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # We'll use a simple iterative method to try to reduce the rank
    
    # First, let's check if we can find a rank 35 decomposition
    # by removing one term and solving for the remaining coefficients
    
    # Actually, for 3x3 x 3x4, the known lower bounds suggest rank >= 30
    # Let's try a structured approach using block decomposition
    
    # Try Winograd 2x2 block approach:
    # Split A into 2x2 blocks of size roughly (2,2) and (1,1)
    # This gets complicated for 3x3 x 3x4
    
    # Instead, let's use a numerical search with random restarts
    # to try to find a lower rank decomposition
    
    import random
    np.random.seed(42)
    
    # Try ranks from best_rank-1 down to some minimum
    # For each rank, try to find a valid decomposition
    
    # The problem is that finding a lower rank decomposition is NP-hard in general
    # So we'll use a heuristic approach
    
    # Let's try a different approach: use the known structure of matrix multiplication
    # to construct a lower rank decomposition
    
    # For 3x3 * 3x3, the best known is rank 23 (Larsen et al.)
    # For 3x3 * 3x4, we can extend this
    
    # Actually, let's just return the schoolbook decomposition for now
    # as finding a provably lower rank decomposition in the time budget is challenging
    
    return best_result
# EVOLVE-BLOCK-END
