"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a better decomposition using structured approaches
    # For 4x4 * 4x9, schoolbook rank is 144
    # Try a rank-100 decomposition based on block structure
    
    # Approach: Use the fact that we can group columns
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Try a decomposition with rank < 144 using linear combinations
    # One known approach: if we can share some partial sums
    
    # For now, use a structured rank-128 approach:
    # Group by pairs of j indices
    
    # Actually, let me try the simplest improvement:
    # Use a rank-144 but verify it's correct, then try to optimize
    
    # Let me implement a search for a better decomposition
    # using a greedy approach with linear algebra
    
    import numpy as np
    
    deadline = time.time() + 120
    
    # Start with schoolbook as baseline
    best_rank = n * m * p
    best_result = None
    
    # Generate schoolbook decomposition
    rank = best_rank
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
    best_result = {"U": U, "V": V, "W": W}
    
    # Verify schoolbook decomposition
    def verify(U, V, W, n, m, p):
        """Check if the decomposition is valid."""
        num_cols = len(U[0]) if U else 0
        if num_cols == 0:
            return False
        # For each (i,j,k), check that sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = delta
        # Actually, the condition is:
        # For all (i,k): C[i,k] = sum_j A[i,j] B[j,k]
        # The decomposition must satisfy: for each output element C[i,k],
        # sum_j A[i,j] B[j,k] = sum_r (sum_j' U[i*m+j'][r] A[i,j']) (sum_k' V[j'*p+k'][r] B[j',k']) W[k*n+i][r]
        # This is complex to verify directly. Let me use a different approach.
        
        # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # and 0 otherwise.
        # The decomposition is valid if for all (a,b,c):
        # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        
        T = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[(i*m+j, j*p+k, k*n+i)] = 1
        
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    s = 0
                    for r in range(num_cols):
                        s += U[a][r] * V[b][r] * W[c][r]
                    expected = T.get((a,b,c), 0)
                    if abs(s - expected) > 1e-9:
                        return False
        return True
    
    # Verify schoolbook
    if not verify(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to find a better decomposition
    # Approach: Use a rank-r decomposition where r < 144
    # This is a hard problem. Let me try a few structured approaches.
    
    # Approach 1: Try rank 100 using a block-diagonal structure
    # Not straightforward for this problem.
    
    # For the time being, return the schoolbook decomposition
    # which is known to be valid.
    
    if time.time() > deadline - 5:
        return best_result
    
    # Try a simple optimization: check if we can merge some columns
    # by finding linear dependencies
    
    # For now, return the schoolbook
    return best_result
# EVOLVE-BLOCK-END
