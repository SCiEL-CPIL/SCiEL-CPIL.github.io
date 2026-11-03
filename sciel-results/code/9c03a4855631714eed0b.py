"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 by 4x5 matrix.
    
    Strategy: Use Winograd-style polynomial evaluation/interpolation to reduce the
    tensor rank below the schoolbook 40.
    
    The matrix multiplication tensor T_{i,j,k} = A_{i,j} * B_{j,k} can be viewed as
    a trilinear form. We can use polynomial techniques:
    
    For 2x4x5:
    - Split the 4 middle dimension into two groups of 2: j1 in {0,1}, j2 in {2,3}
    - For each group, use a 2x2x5 sub-problem
    - Use polynomial evaluation to reduce the rank
    
    Key insight: For multiplying a 2x2 matrix by a 2xp matrix, we can use
    the fact that (a+b)(c+d) = ac + ad + bc + bd can be computed with 3 multiplications
    (Strassen-style).
    
    Let's build this step by step:
    
    A is 2x4, B is 4x5, C is 2x5.
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    Split j into j1={0,1} and j2={2,3}:
    C[i,k] = sum_{j1=0}^{1} A[i,j1]*B[j1,k] + sum_{j2=0}^{1} A[i,j2+2]*B[j2+2,k]
    
    For the first part (2x2x5):
    Let a0 = A[i,0], a1 = A[i,1], b0_k = B[0,k], b1_k = B[1,k]
    Part1[i,k] = a0*b0_k + a1*b1_k
    
    This is already rank 2 per (i,k) pair, but we can do better.
    
    Actually, let's think of it differently. The tensor has dimensions 2x4x5.
    We want to find the minimal rank decomposition.
    
    Using the approach from Winograd: we can evaluate polynomials at specific points
    and interpolate to reduce the number of multiplications.
    
    For the 2x4x5 case, let's try a specific construction:
    
    Let's use the fact that we can combine some terms. Consider:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1] + A[0,3]*B[3,1]
    ...
    
    A known lower bound for the rank of the 2x4x5 matrix multiplication tensor
    can be estimated. The schoolbook rank is 40.
    
    Let's try to find a rank-36 or lower decomposition using polynomial methods.
    
    Actually, let's use a simpler but effective approach:
    Use the 2x2x2 Strassen decomposition for parts of the problem.
    
    Split the 4x5 B matrix into four 2x5 blocks, or the 2x4 A into 2x2 blocks.
    
    Let's try: Split A into A1 (2x2) and A2 (2x2), and B into B1 (2x5) and B2 (2x5).
    C = A1*B1 + A2*B2
    
    Each 2x2x5 multiplication has rank at most 10 (schoolbook), but can be done in rank 8
    using Strassen-like techniques for the 2x2 part.
    
    For 2x2x5: C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k]
    This is a 2x2x5 tensor. The schoolbook rank is 4*5=20? No, it's 2*2*5=20.
    But we can do better.
    
    Actually, for 2x2xp, the rank is known to be 2p-2 for p>=2 (using Strassen).
    For p=5: rank = 2*5-2 = 8.
    
    So total rank = 8 + 8 = 16? That seems too good. Let me reconsider.
    
    Wait, the 2x2xp tensor has entries T_{i,j,k} = A_{i,j} * B_{j,k}.
    The rank of this tensor... For 2x2x2, the rank is 7 (Strassen).
    For 2x2x3, the rank is 10? Or 9?
    
    Let me use a different approach. Let's just implement a careful search or use
    known results.
    
    Actually, let's just return the schoolbook decomposition for now, but with a
    deadline guard. The task says to find rank-minimal, but if we can't find a
    better one, we return the best we have.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
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
    
    # Try to improve using Strassen-like decomposition
    # For 2x4x5, we can try to use polynomial evaluation
    
    # Let's attempt a rank-32 decomposition using Winograd-style methods
    # This is a heuristic attempt
    
    if time.time() < deadline - 5:
        # Try a specific construction
        # Split the middle dimension into two 2-dim parts
        # Use 2x2x5 subproblems with rank 8 each (if possible)
        
        # For 2x2x5: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        # We can use: Let x = A[i,0]+A[i,1], y = B[0,k]+B[1,k]
        # Then A[i,0]*B[0,k] + A[i,1]*B[1,k] = (x*y - A[i,0]*B[1,k] - A[i,1]*B[0,k]) / 1
        # This doesn't help directly.
        
        # Use Strassen for the 2x2 part:
        # m1 = (A[i,0]+A[i,1]) * B[0,k]
        # m2 = A[i,1] * (B[0,k]+B[1,k])
        # m3 = A[i,0] * B[1,k]
        # Then A[i,0]*B[0,k] = m1 - m3
        # And A[i,1]*B[1,k] = m2 - m1
        # So we need 3 multiplications per (i,k) pair instead of 2? No, that's worse.
        
        # Actually for 2x2xp, the optimal rank is 2p-1 for p=2 (Strassen gives 3, but 2x2x2 has rank 7)
        # I'm confusing myself. Let's just return schoolbook.
        
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
