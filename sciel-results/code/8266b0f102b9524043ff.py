"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_fallback = [[0] * rank_school for _ in range(n * m)]
    V_fallback = [[0] * rank_school for _ in range(m * p)]
    W_fallback = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][col] = 1
                V_fallback[j * p + k][col] = 1
                W_fallback[k * n + i][col] = 1
                col += 1
    
    # Build the matrix multiplication tensor T[i,j,k] = delta[i,j,k] 
    # where T[i,j,k] = 1 if i,j,k correspond to a valid multiplication term
    # The tensor has indices: A[i,j], B[j,k] -> C[i,k]
    # T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Try randomized search for lower rank
    best_rank = rank_school
    best_U, best_V, best_W = U_fallback, V_fallback, W_fallback
    
    # Strategy: Use Strassen-like block decomposition ideas
    # For 4x5 * 5x5, we can try various rank targets
    
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # Upper bound from schoolbook: 100
    # Try to find something better using structured random search
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == (1 if j matches else 0)
        # Actually for matmul: C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor is T[i,j,k] where the only nonzero entries are when the indices align
        # For A[i,j] * B[j,k] -> contributes to C[i,k]
        # So T[i,j,k] = 1 for all valid (i,j,k) tuples, and the decomposition must satisfy:
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try a few structured approaches
    # Approach 1: Block Strassen-like for the 4x5 * 5x5 case
    # We can partition A as [A1; A2] where A1 is 2x5, A2 is 2x5
    # B as [B1 B2] where B1 is 5x2, B2 is 5x3 (but this doesn't divide evenly)
    
    # Better: Try random initialization with optimization
    # Use the fact that we need integer/half-integer entries
    
    # Let's try a simple approach: assume rank R and try to find U,V,W
    # such that the tensor equation holds. This is a system of polynomial equations.
    
    # For now, let's try some small ranks with random search
    for target_rank in [30, 35, 40, 45, 50]:
        if time.time() > deadline:
            break
        
        # Random initialization with small integers/half-integers
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random small values from {-1, 0, 1, 0.5, -0.5}
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, target_rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, target_rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, target_rank))
            
            # Convert to lists for verification
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify(U_list, V_list, W_list, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_list, V_list, W_list
                break
    
    # Return the best found
    if best_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
# EVOLVE-BLOCK-END
