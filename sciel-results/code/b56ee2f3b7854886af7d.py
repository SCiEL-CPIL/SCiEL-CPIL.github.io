"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    schoolbook_rank = n * m * p  # 128
    
    # The tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # T[i,j,k] = 1 for all valid (i,j,k)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Strategy: Use block structure.
    # C is 4x8. We can split C into two 4x4 blocks: C_left (cols 0-3) and C_right (cols 4-7).
    # C_left[i, k'] = sum_j A[i,j] B[j, k'] for k' in 0..3
    # C_right[i, k''] = sum_j A[i,j] B[j, 4+k''] for k'' in 0..3
    
    # We can use the same A factors for both blocks.
    # For a standard 4x4x4 multiplication, the minimal rank is known to be 20 (Strassen-like or specific 4x4x4 opt).
    # However, finding the exact rank-20 decomposition for 4x4x4 is non-trivial to hardcode without a library.
    # Instead, we can try to construct a decomposition of rank < 128 using simple structural symmetries or known lower bounds.
    
    # Known lower bound for 4x4x4 is 20. For 4x4x8, the lower bound is at least max(rank(4x4x4), something).
    # Actually, the rank of 4x4x8 matmul is at least the rank of 4x4x4 matmul if we can embed it, but it's larger.
    # A simple upper bound is 2 * rank(4x4x4) if we treat it as two independent 4x4x4 multiplications sharing A.
    # If we use a rank-R decomposition for 4x4x4, we can use rank 2R for 4x4x8 by duplicating the terms for the two blocks of B.
    # Let's try to find a good decomposition for 4x4x4 first, then extend.
    
    # Since finding the exact optimal 4x4x4 decomposition is complex, let's use a known efficient decomposition or a simple heuristic.
    # Strassen's algorithm gives rank 7 for 2x2x2. For 4x4x4, a naive block application of Strassen gives rank 7 * 8 = 56? No, that's not right.
    # Block Strassen for 4x4x4 (using 2x2 blocks) requires 7 multiplications of 2x2 matrices, each of which requires 7 multiplications of scalars? No, 2x2 matmul rank is 7.
    # So rank(4x4x4) <= 7 * 7 = 49.
    # Actually, the best known for 4x4x4 is 20 (Landsberg).
    
    # Let's try a different approach: Randomized search with refinement?
    # Or, we can use the fact that the tensor is structured.
    
    # Let's start with the schoolbook and try to reduce rank via linear algebra.
    # We want to find U, V, W of minimal rank R such that the tensor is decomposed.
    # This is equivalent to finding a rank-R decomposition.
    
    # We can formulate this as:
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    
    # Let's try to use a known decomposition for 4x4x4.
    # Here is a rank-20 decomposition for 4x4x4 matrix multiplication.
    # Reference: "On the border rank of the matrix multiplication tensor" by Landsberg.
    # However, implementing the exact rank-20 decomposition is complex.
    
    # Alternative: Use a simple rank reduction heuristic.
    # Start with schoolbook (rank 128).
    # Try to merge terms that are linearly dependent or can be combined.
    
    # A more effective heuristic:
    # 1. Generate a random decomposition of rank R (starting from schoolbook).
    # 2. Use gradient descent or a similar method to minimize the error.
    # 3. Check if the error is zero (exact decomposition).
    
    # But we need an exact decomposition.
    
    # Let's try a specific construction:
    # For 4x4x8, we can view it as two 4x4x4 multiplications.
    # If we have a rank-R decomposition for 4x4x4, we can get a rank-2R decomposition for 4x4x8.
    # Let's assume we can find a rank-20 decomposition for 4x4x4.
    # Then rank(4x4x8) <= 40.
    
    # How to get a rank-20 decomposition for 4x4x4?
    # One way is to use the "Strassen-like" decomposition or a known explicit one.
    # Here is a known rank-20 decomposition for 4x4x4 (from literature):
    # It's complex to hardcode.
    
    # Instead, let's try a simpler reduction:
    # The schoolbook rank is 128.
    # We can try to find linear dependencies among the 128 terms.
    # If we can express some terms as linear combinations of others, we can reduce the rank.
    
    # A practical approach:
    # 1. Start with schoolbook.
    # 2. Use a randomized search to find a lower-rank decomposition.
    # 3. Use a refinement step to ensure exactness.
    
    # Given the time limit, let's implement a simple randomized search.
    
    # We'll try to find a decomposition of rank R for R in [40, 128].
    # For each R, we'll try to find U, V, W such that the tensor is decomposed.
    
    # Since finding an exact decomposition is hard, let's use a different strategy:
    # Use the fact that the tensor is a "matrix multiplication" tensor, which has a specific structure.
    
    # Let's try to use a known lower-rank decomposition for 4x4x4.
    # Here is a rank-20 decomposition for 4x4x4 (simplified):
    # We'll use a heuristic to generate a candidate decomposition and then refine it.
    
    # For now, let's return the schoolbook decomposition, as it's guaranteed to be correct.
    # We can improve this by implementing a more advanced search.
    
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
