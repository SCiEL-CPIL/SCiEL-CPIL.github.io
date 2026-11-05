"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # U shape: (n*m, rank) = (12, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (18, rank)
    # W[k*n+i, c] corresponds to output C[i,k]
    
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Strategy: Split p=6 into three blocks of size 2.
    # For each block of size 2, we have a 3x4 * 4x2 multiplication.
    # We can further decompose this using Strassen-like methods.
    
    # Actually, let's use a more direct approach:
    # The tensor for 3x4 * 4x6 can be viewed as 3 independent 3x4 * 4x2 problems
    # (one for each pair of columns k=0,1; k=2,3; k=4,5).
    
    # For 3x4 * 4x2, we can use a decomposition.
    # Let's think about this differently.
    
    # For 3x4 * 4x2:
    # C[i,0] = sum_j A[i,j]*B[j,0]
    # C[i,1] = sum_j A[i,j]*B[j,1]
    
    # We can group by i (3 rows of A) and use the 4x2 part.
    # For each i, we need to compute a 4x2 matrix product contribution.
    
    # Actually, let me use a known result: M(3,4,2) has rank at most 11.
    # So 3 * 11 = 33 would be one approach, but we can do better by sharing.
    
    # Better approach: Use the fact that we can split the 4 middle dimension.
    # Or use a block decomposition.
    
    # Let me try a direct construction with Strassen for 2x2x2 blocks.
    
    # Split m=4 into two blocks of 2, and p=6 into three blocks of 2.
    # This gives us a 3x2 * 2x2 problem for each (row block, col block) pair.
    # Wait, that's not quite right either.
    
    # Let me use a cleaner approach:
    # The matrix multiplication 3x4 * 4x6 can be seen as:
    # For each output element C[i,k], we compute sum_{j=0}^{3} A[i,j]*B[j,k]
    
    # Group j into two groups: {0,1} and {2,3}
    # Group k into three groups: {0,1}, {2,3}, {4,5}
    
    # For each (j-group, k-group), we have a 3x2 * 2x2 subproblem.
    # There are 2 * 3 = 6 such subproblems.
    # Each 3x2 * 2x2 has rank at most... let me think.
    # M(3,2,2): The tensor has 3*2*2 = 12 entries.
    # Lower bound: max(3*2, 2*2, 3*2) = max(6,4,6) = 6.
    # Upper bound: schoolbook is 3*2*2 = 12.
    # Can we do better? 
    
    # Actually, let me just use a systematic construction.
    
    # Approach: Use the "border rank" or a known decomposition.
    
    # Let me try rank = 24 first (which is m*p = 24, the maximum dimension product).
    # If that doesn't work, try higher ranks.
    
    # Actually, the simplest valid decomposition is the schoolbook one with rank = n*m*p = 72.
    # Let me start there and try to reduce.
    
    # For now, let me implement a working solution and then optimize.
    
    # Schoolbook: rank = 72
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        for a in range(n*m):
            i = a // m
            j = a % m
            for b in range(m*p):
                j2 = b // p
                k = b % p
                for c in range(p*n):
                    k2 = c // n
                    i2 = c % n
                    expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                    s = 0.0
                    for d in range(U.shape[1]):
                        s += U[a,d] * V[b,d] * W[c,d]
                    if abs(s - expected) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        pass  # Should not happen for schoolbook
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
