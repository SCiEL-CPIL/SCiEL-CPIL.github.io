"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # We'll use a block decomposition approach
    # Strategy: Use Strassen-like decompositions for sub-blocks
    # 
    # For 2x5x6, we can try various block splits:
    # Option 1: 2x2 + 2x2 + 2x1 (columns of B split)
    # Option 2: Use known rank bounds
    #
    # Known results:
    # - Rank(2x2x2) = 7 (Strassen)
    # - Rank(2x2x1) = 4 (trivial, can't do better)
    # - Rank(2x1x2) = 2
    # - Rank(1x2x2) = 2
    #
    # Let's try a decomposition that achieves rank 40:
    # Split the 5 columns into groups and use Strassen where possible
    
    # Approach: Direct construction using block Strassen
    # 
    # Actually, let's think about this differently.
    # The trivial rank is 2*5*6 = 60.
    # 
    # Let's try to construct a rank-40 decomposition:
    # Split B into B1 (2x2), B2 (2x2), B3 (1x2) along columns
    # Then C = A1*B1 + A2*B2 + A3*B3 where A1,A2 are 2x2 and A3 is 2x1
    #
    # But this doesn't directly help because A is shared.
    #
    # Better approach: Use the fact that we can do Strassen on 2x2x2 blocks.
    # For 2x5x6:
    # - Split into five 2x1x6 blocks: rank = 5 * 2 = 10? No, that's not right.
    #   Each 2x1x6 block has rank 6 (trivial), so 5*6 = 30.
    #
    # Alternative: Group columns
    # - 2x2x6 can be done with rank 14 (using Strassen on each 2x2x2)
    #   Actually 2x2x6: split into three 2x2x2, each rank 7, total 21
    #   But we can do better: 2x2x6 has rank 14 (known result)
    #
    # Let me just use a clean direct construction.
    #
    # For rank 40:
    # Use the following block structure:
    # Think of the middle index j (0..4) being paired up
    #
    # Actually, let me try the simplest correct approach first:
    # Trivial rank 60, then try to reduce.
    
    # Let's try rank 40 construction:
    # 
    # The key insight: for 2x5x6, we can use the following:
    # - Pair up 4 of the 5 middle indices into two pairs
    # - For each pair, use Strassen on the 2x2x(sub) blocks
    #
    # More carefully:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Let's group j into: {0,1}, {2,3}, {4}
    # For group {0,1}: C += A[:,{0,1}] @ B[{0,1},:]  (2x2 times 2x6)
    # For group {2,3}: C += A[:,{2,3}] @ B[{2,3},:]  (2x2 times 2x6)  
    # For group {4}:   C += A[:,{4}]   @ B[{4},:]    (2x1 times 1x6)
    #
    # Rank(2x2x6): We can do this in rank 14
    #   Split k into {0,1},{2,3},{4,5}: three 2x2x2 blocks, each rank 7 = 21
    #   Or better: 2x2x6 has known rank 14
    #
    # Rank(2x1x6): rank 6 (trivial)
    #
    # Total: 14 + 14 + 6 = 34? That seems too good.
    #
    # Wait, rank(2x2x6) is actually 14? Let me verify.
    # 2x2x2 has rank 7. 2x2x6 = three 2x2x2 blocks = 21 trivially.
    # Can we do 2x2x6 in rank 14? 
    # C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, j in {0,1}, k in {0..5}
    # This is a 2x2x6 tensor. The rank is at most 2*6=12 (by fixing i)
    # or at most 2*2*6=24 trivially.
    # Actually rank(2x2x6) <= min(2*6, 2*6, 2*2*6) but with Strassen-like:
    # We can do it in rank 14 by splitting k into pairs.
    #
    # Hmm, let me just be safe and use rank 40.
    #
    # Let me construct rank 40:
    # - Two 2x2x6 blocks (for j pairs {0,1} and {2,3}), each in rank 14
    # - One 2x1x6 block (for j=4) in rank 6
    # Total: 14 + 14 + 6 = 34
    #
    # But I need to actually construct the 2x2x6 rank-14 decomposition.
    #
    # For 2x2x6:
    # A is 2x2, B is 2x6, C is 2x6
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # Split k into {0,1}, {2,3}, {4,5}
    # For each pair of k values, we have a 2x2x2 sub-problem
    # Using Strassen on each: rank 7 per pair, total 21
    #
    # But we can share: for 2x2x6, the rank is actually 14
    # because we can write:
    # Let B = [b0, b1] where b0, b1 are 1x6 vectors (rows of B)
    # C[0,:] = A[0,0]*b0 + A[0,1]*b1
    # C[1,:] = A[1,0]*b0 + A[1,1]*b1
    #
    # This is like computing a 2x2 matrix times a 2x6 matrix.
    # We can use: C = A @ B
    # 
    # Using Strassen-like for 2x2 x 2x6:
    # Think of B as six 2x1 vectors, or as a 2x6 matrix.
    # 
    # Actually, the simplest: rank(2x2x6) = 12
    # Because: for each of the 6 columns of B, we need to compute
    # [A[0,0]*B[0,k] + A[0,1]*B[1,k], A[1,0]*B[0,k] + A[1,1]*B[1,k]]
    # This is a 2x2 matrix-vector multiply for each k, which takes rank 4
    # But we can share the A factors...
    #
    # Actually rank(2x2x6) = 12: just use 4 terms per column of k, but share:
    # No wait. Let me think again.
    #
    # C[i,k] = sum_j A[i,j] B[j,k]
    # For fixed i, this is a linear combination of the two rows of B.
    # The tensor is sum_{i,j,k} A[i,j] B[j,k] C[i,k]
    # = sum_k (sum_{i,j} A[i,j] B[j,k] C[i,k])
    #
    # For each k, the inner tensor is A @ B[:,k] which is a 2x2x1 tensor of rank 4.
    # But across k values, we can share the A factors.
    #
    # Total rank = 6 * 2 = 12? No...
    # 
    # Let me just use rank 21 for 2x2x6 (three Strassen 2x2x2 blocks)
    # and rank 6 for 2x1x6.
    # Total: 21 + 21 + 6 = 48. Still better than 60.
    #
    # Even better: use rank 14 for 2x2x6 if I can construct it.
    #
    # Let me try: for 2x2x6, use the following 14-term decomposition:
    # 
    # Actually, I'll just go with a clean rank-48 construction and verify.
    #
    # Let me try yet another approach: 
    # For 2x5x6, use rank 40 by the following:
    # 
    # Known: rank(2x5x6) <= 40 (from literature on matrix multiplication ranks)
    #
    # Let me construct rank 40 explicitly.
    
    # I'll use a different strategy: direct rank-40 construction
    # 
    # For 2x5x6:
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    #
    # Use the following 40 terms:
    # Group 1 (14 terms): handle j=0,1 with k=0,1 using Strassen 2x2x2
    # Group 2 (14 terms): handle j=2,3 with k=2,3 using Strassen 2x2x2
    # Group 3 (6 terms): handle j=4 with all k=0..5 trivially
    #
    # Wait, that's 14+14+6 = 34, but the Strassen 2x2x2 only handles 2x2x2.
    # For j=0,1 and k=0,1: that's A[:,{0,1}] @ B[{0,1},{0,1}] -> 2x2 result
    # But we need the FULL 2x6 result, not just 2x2.
    #
    # I think I'm overcomplicating this. Let me just do:
    # 
    # Rank 40 construction:
    # For each of the 6 k-values, compute the 2x5 dot product with B[:,k]
    # The 2x5 times 5x1 multiply has rank at most 10 (trivial)
    # But we can share A factors across k values.
    #
    # Actually, the simplest correct approach:
    # Total rank = n*m*p = 60 (trivial)
    #
    # Let me just implement the trivial rank-60 and then try to reduce.
    
    # TRIVIAL CONSTRUCTION (rank 60):
    rank = n * m * p  # 60
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
    
    # Verify: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For the trivial case, each term r corresponds to (i,j,k):
    # U[i*m+j, r] = 1 only when the (i,j) part matches
    # V[j*p+k, r] = 1 only when the (j,k) part matches
    # W[k*n+i, r] = 1 only when the (k,i) part matches
    # So C[i,k] gets contribution A[i,j]*B[j,k] from term (i,j,k). Correct.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
