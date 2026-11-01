"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # Try DFT-based decomposition for the p=9 dimension
    # The key insight: if we can decompose using DFT basis, we might reduce rank
    # For 2x2x9, schoolbook is 36. Let's try to find a lower rank decomposition.
    
    # Strategy: Use the fact that 9 = 3*3 and try block decomposition
    # Or use DFT over Z_9 to exploit periodicity
    
    # First, let's try a simple improvement: block decomposition
    # Split p=9 into blocks and use efficient multiplication for each block
    
    # For 2x2 matrices, Strassen gives rank 7 instead of 8
    # For 2x2x1, rank is 4 (schoolbook)
    
    # Let's try: decompose 2x2x9 as combination of Strassen blocks
    # This is complex, so let's first try a DFT approach
    
    # DFT approach: use 9th roots of unity
    # omega = exp(2*pi*i/9)
    # But we need exact rational/integer coefficients
    
    # Alternative: use the cyclic group structure directly
    # The tensor T_{i,j,k} = e_{i,j} * e_{j,k} * e_{k,i} (simplified)
    
    # Let's try a rank reduction by sharing factors across k values
    # For fixed (i,j), the k-dependence is linear, so we might combine k's
    
    # Actually, let's try a different structural approach:
    # Use the fact that 9 is composite (9=3*3) and do a two-level decomposition
    
    # Level 1: Split k into k1 + 3*k2 where k1,k2 in {0,1,2}
    # This gives us a 2x2x3 x 2x2x3 structure
    
    # For now, let's return the schoolbook solution as baseline
    # and add a deadline guard
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # Return schoolbook as fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition using linear algebra
    # The matrix multiplication tensor has a known structure
    # For 2x2x2, the minimal rank is 7 (Strassen)
    # For 2x2x9, we can try to combine Strassen-like tricks
    
    # Let's attempt a constructive approach:
    # Group the 9 k-values into 3 groups of 3
    # For each group, use a 2x2x3 decomposition
    
    # 2x2x3 schoolbook rank is 12
    # But maybe we can do better by sharing across groups
    
    # Actually, let's try a direct optimization:
    # Assume rank r, and try to find U, V, W with integer/half-integer entries
    
    # This is a hard problem, so let's use a heuristic:
    # Start with schoolbook and try to merge columns that are linearly dependent
    
    # For now, return schoolbook with a small improvement attempt
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to reduce rank by finding linear dependencies
    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The decomposition is valid if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j} * delta_{j,k} * delta_{k,i}
    # Actually, the tensor is T[i,j,k] = 1 if the indices are consistent, else 0
    # For matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 for all i,j,k, and the decomposition gives
    # (A*B)[i,k] = sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
    # Wait, that's not right. Let me reconsider.
    
    # The correct decomposition: 
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed appropriately
    # Actually, the standard form is:
    # (A*B)[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
    # No, that's still not right.
    
    # Let me use the correct tensor decomposition form:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where T[i,j,k] = 1 for all i,j,k (since it's the identity tensor for matmul)
    # Actually, T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in (A*B)[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k
    
    # The decomposition is:
    # 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i,j,k
    
    # This is a multiplicative identity, which is hard to satisfy with rank < n*m*p
    
    # For now, return the schoolbook solution
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
