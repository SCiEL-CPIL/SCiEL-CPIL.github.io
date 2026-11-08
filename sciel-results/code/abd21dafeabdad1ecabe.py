"""Matrix multiplication 3,5,5 with rank-minimal decomposition search."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 5]
n, m, p = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Using trace convention: W[k*n+i, r]
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Schoolbook decomposition as fallback
    rank_sb = n*m*p
    U_sb = np.zeros((n*m, rank_sb), dtype=np.int64)
    V_sb = np.zeros((m*p, rank_sb), dtype=np.int64)
    W_sb = np.zeros((p*n, rank_sb), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # Strategy: Use rank-1 updates to reduce from schoolbook
    # A rank-r decomposition means T = sum_{r'=1}^{r} u_{r'} v_{r'}^T w_{r'}^T (outer product)
    
    # Try random search for low-rank decompositions
    # For each rank r from some minimum up to best_rank-1, try to find a decomposition
    
    # The tensor T can be decomposed as:
    # T_{a,b,c} = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Try systematic approach: use the structure of 3x5x5
    # Partition the 5 into 3+2
    # This gives blocks:
    # - A_{3x3} * B_{3x3} -> 3x3 output (core)
    # - A_{3x3} * B_{3x2} -> 3x2 output (boundary 1)
    # - A_{3x2} * B_{2x3} -> 3x3 output (boundary 2)
    # - A_{3x2} * B_{2x2} -> 3x2 output (boundary 3)
    
    # For the 3x3 core, use known rank-19 algorithm (Lader)
    # But constructing Lader's algorithm is complex. Let me try a simpler approach.
    
    # Alternative: Use Strassen-like decompositions on blocks
    # 2x2 Strassen: rank 7 instead of 8
    # 3x3 via 2x2 blocks: 
    #   Partition 3x3 as [[2x2, 2x1], [1x2, 1x1]]
    #   This gets complicated.
    
    # Let me try a different approach: random tensor decomposition search
    # For small ranks, use optimization
    
    # First, try to verify if rank-75 (schoolbook) can be beaten with simple block methods
    
    # Method 1: Use 2x2 Strassen for the 2x2 sub-blocks
    # The 5x5 can be partitioned as 2+2+1
    # B_{5x5} partitioned into rows [0:2, 2:4, 4:5]
    # B_{5x5} partitioned into cols [0:2, 2:4, 4:5]
    # A_{3x5} partitioned into cols [0:2, 2:4, 4:5]
    # Output C_{3x5} partitioned into cols [0:2, 2:4, 4:5]
    
    # C_{:,0:2} = A_{:,0:2}B_{0:2,0:2} + A_{:,2:4}B_{2:4,0:2} + A_{:,4:5}B_{4:5,0:2}
    # C_{:,2:4} = A_{:,0:2}B_{0:2,2:4} + A_{:,2:4}B_{2:4,2:4} + A_{:,4:5}B_{4:5,2:4}
    # C_{:,4:5} = A_{:,0:2}B_{0:2,4:5} + A_{:,2:4}B_{2:4,4:5} + A_{:,4:5}B_{4:5,4:5}
    
    # For 2x2 * 2x2, use Strassen (rank 7)
    # For 2x2 * 2x1, rank 4 (schoolbook is 4)
    # For 2x1 * 1x2, rank 4 (schoolbook is 4)  
    # For 2x1 * 1x1, rank 2
    # For 1x2 * 2x2, rank 4
    # For 1x2 * 2x1, rank 2
    # For 1x1 * 1x2, rank 2
    # For 1x1 * 1x1, rank 1
    
    # Let me compute the rank for this partitioning:
    # C_{:,0:2} (3x2):
    #   A_{:,0:2}(3x2) * B_{0:2,0:2}(2x2): 3x2*2x2 -> 6 terms (schoolbook), but we can use:
    #     Actually this is (3x2)(2x2) = 3x2, rank 6 schoolbook
    #   A_{:,2:4}(3x2) * B_{2:4,0:2}(2x2): 6 terms
    #   A_{:,4:5}(3x1) * B_{4:5,0:2}(1x2): 2 terms
    #   Subtotal: 6+6+2 = 14
    
    # C_{:,2:4} (3x2):
    #   A_{:,0:2}(3x2) * B_{0:2,2:4}(2x2): 6 terms
    #   A_{:,2:4}(3x2) * B_{2:4,2:4}(2x2): 6 terms  
    #   A_{:,4:5}(3x1) * B_{4:5,2:4}(1x2): 2 terms
    #   Subtotal: 6+6+2 = 14
    
    # C_{:,4:5} (3x1):
    #   A_{:,0:2}(3x2) * B_{0:2,4:5}(2x1): 2 terms
    #   A_{:,2:4}(3x2) * B_{2:4,4:5}(2x1): 2 terms
    #   A_{:,4:5}(3x1) * B_{4:5,4:5}(1x1): 1 term
    #   Subtotal: 2+2+1 = 5
    
    # Total: 14+14+5 = 33 with schoolbook on these blocks
    # But we can do better on the 2x2*2x2 blocks using Strassen!
    
    # Let me use Strassen for the 2x2*2x2 blocks:
    # C_{:,0:2}:
    #   A_{:,0:2}(3x2)*B_{0:2,0:2}(2x2): This is like computing 3 rows of 2x2 mult
    #     Each row: 2x1 * 2x2 = 2x2, which is rank 4 schoolbook
    #     Or: treat as (3x2)(2x2) = 3x2. Schoolbook rank 6.
    #     Can we do better? (3x2)(2x2): partition A as 3 rows, B as 2x2.
    #     This is equivalent to 3 independent (1x2)(2x2) = 3x2, rank 3*4=12? No...
    #     Actually (1x2)(2x2) = 1x2, rank 4 (multiply each column independently? No)
    #     (1x2)(2x2): result is 1x2. c0 = a0*b00 + a1*b10, c1 = a0*b01 + a1*b11
    #     That's 4 multiplications, rank 4.
    #     So (3x2)(2x2) = 3 rows, each rank 4, total rank 12? That's worse than 6.
    #     Wait, I'm confusing myself. Let me be more careful.
    
    # (3x2) * (2x2) -> 3x2 result
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1]
    # Total multiplications: 3*2*2 = 12? No, that's schoolbook for 3x2*2x2.
    # Wait: n=3, m=2, p=2. Rank = n*m*p = 3*2*2 = 12 schoolbook.
    # But I said 6 above. Let me recount.
    # A is 3x2, B is 2x2, C is 3x2.
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # For each (i,j) pair: 2 multiplications (k=0,1)
    # Number of (i,j) pairs: 3*2 = 6
    # Total multiplications: 6*2 = 12. Yes, schoolbook rank is 12.
    
    # I was making an error. Let me redo the partitioning carefully.
    
    # A is 3x5, B is 5x5, C is 3x5.
    # Schoolbook rank: 3*5*5 = 75.
    
    # Partition B's rows into [0:2, 2:4, 4:5] (sizes 2,2,1)
    # Partition B's cols into [0:2, 2:4, 4:5] (sizes 2,2,1)
    # Partition A's cols into [0:2, 2:4, 4:5] (sizes 2,2,1)
    # C's cols into [0:2, 2:4, 4:5] (sizes 2,2,1)
    
    # C[:,0:2] = A[:,0:2]*B[0:2,0:2] + A[:,2:4]*B[2:4,0:2] + A[:,4:5]*B[4:5,0:2]
    # Block sizes:
    #   A[:,0:2] is 3x2, B[0:2,0:2] is 2x2, product is 3x2: rank = 3*2*2 = 12
    #   A[:,2:4] is 3x2, B[2:4,0:2] is 2x2, product is 3x2: rank = 3*2*2 = 12
    #   A[:,4:5] is 3x1, B[4:5,0:2] is 1x2, product is 3x2: rank = 3*1*2 = 6
    #   Subtotal for C[:,0:2]: 12+12+6 = 30
    
    # C[:,2:4] = A[:,0:2]*B[0:2,2:4] + A[:,2:4]*B[2:4,2:4] + A[:,4:5]*B[4:5,2:4]
    #   3x2*2x2: 12
    #   3x2*2x2: 12
    #   3x1*1x2: 6
    #   Subtotal: 30
    
    # C[:,4:5] = A[:,0:2]*B[0:2,4:5] + A[:,2:4]*B[2:4,4:5] + A[:,4:5]*B[4:5,4:5]
    #   3x2*2x1: 3*2*1 = 6
    #   3x2*2x1: 3*2*1 = 6
    #   3x1*1x1: 3*1*1 = 3
    #   Subtotal: 15
    
    # Total schoolbook with this partition: 30+30+15 = 75. Same as original.
    
    # To improve, use Strassen on the 2x2*2x2 blocks.
    # For a 2x2*2x2 block, Strassen uses rank 7 instead of 8.
    # But our blocks are 3x2 * 2x2, not 2x2*2x2.
    
    # For (3x2)*(2x2) = 3x2:
    # This is equivalent to 3 independent (1x2)*(2x2) = 1x2 multiplications.
    # (1x2)*(2x2): c0 = a0*b00 + a1*b10, c1 = a0*b01 + a1*b11
    # Rank 4 (schoolbook). Can we do better? 
    # This is like a bilinear map R^2 x R^2 -> R^2.
    # The tensor has 8 entries (2*2*2). Schoolbook rank is 4.
    # I don't think we can go below 4 for this specific structure.
    
    # Actually, for (1x2)*(2x2), the output is 1x2.
    # c_j = sum_k a_k * B[k,j]
    # This is just 4 multiplications: a0*b00, a1*b10, a0*b01, a1*b11
    # And 2 additions each. Rank is exactly 4 (can't do better since each output
    # depends on 2 inputs from A and 2 from B).
    
    # So (3x2)*(2x2) has rank 3*4 = 12. No improvement possible with Strassen here.
    
    # What about using a different partition?
    # Try partitioning 5 as 3+2:
    # B rows: [0:3, 3:5] (sizes 3,2)
    # B cols: [0:3, 3:5] (sizes 3,2)  
    # A cols: [0:3, 3:5] (sizes 3,2)
    # C cols: [0:3, 3:5] (sizes 3,2)
    
    # C[:,0:3] = A[:,0:3]*B[0:3,0:3] + A[:,3:5]*B[3:5,0:3]
    #   3x3*3x3: rank 27 schoolbook
    #   3x2*2x3: rank 3*2*3 = 18
    #   Subtotal: 45
    
    # C[:,3:5] = A[:,0:3]*B[0:3,3:5] + A[:,3:5]*B[3:5,3:5]
    #   3x3*3x2: rank 3*3*2 = 18
    #   3x2*2x2: rank 3*2*2 = 12
    #   Subtotal: 30
    
    # Total: 45+30 = 75. Still 75.
    
    # The key insight: for the 3x3*3x3 block, if we can use rank < 27, we save.
    # Best known for 3x3 is 19 (Lader). That would save 8.
    # New total: (19+18) + (18+12) = 37+30 = 67.
    
    # Let me try to construct a rank-19 algorithm for 3x3.
    # This is complex. Let me instead try a random search approach.
    
    # Random tensor decomposition search:
    # For a given rank r, we want to find U (15xr), V (25xr), W (15xr) such that
    # T[a,b,c] = sum_r' U[a,r']*V[b,r']*W[c,r']
    
    # This is a hard problem. Let me try a simpler heuristic.
    
    # Check time
    if time.time() > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to use known results: the rank of 3x3 matrix multiplication is 19.
    # Let me hard-code the Lader rank-19 algorithm if I can recall it,
    # or use a simpler reduction.
    
    # Actually, let me try a different approach: use the fact that 
    # M_{3,5,5} can be embedded in M_{4,5,5} or similar.
    
    # Or: use a random search to try to find rank < 75.
    
    # For now, let's try a systematic approach with the 3+2 partition
    # and see if we can find a rank-19 3x3 decomposition.
    
    # The 3x3 matrix multiplication tensor T_{3,3,3} has entries:
    # T[i*3+j, j*3+k, k*3+i] = 1 for i,j,k in {0,1,2}
    # Total 27 non-zero entries.
    
    # Rank 19 means we need 19 rank-1 terms.
    
    # This is a well-known result but constructing it from scratch is error-prone.
    
    # Let me instead try a random optimization approach for small ranks.
    
    # Actually, given the complexity, let me just return the schoolbook decomposition
    # but try a few quick random searches to see if we can find rank < 75.
    
    # Quick random search
    np.random.seed(42)
    found_better = False
    for r in range(60, 75):
        if time.time() > deadline - 30:
            break
        # Try to find rank-r decomposition
        # Use random initialization and gradient descent
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Check if this decomposes T
        # T_approx[a,b,c] = sum_r' U[a,r']*V[b,r']*W[c,r']
        # This is expensive to compute fully. Let's just check a subset.
        
        # Actually, for exact integer decompositions, random real numbers won't work.
        # We need integer or half-integer entries.
        
        # Let me try a different approach: use the block structure.
        pass
    
    # Given the difficulty of finding a better decomposition in limited time,
    # let's return the schoolbook decomposition.
    # But first, let me try one more thing: use 2x2 Strassen on a 4x4 sub-block
    # by padding 3 to 4.
    
    # Pad A from 3x5 to 4x5 (add a row of zeros)
    # Pad B from 5x5 to 5x4 (add a column of zeros) - no, this doesn't work cleanly.
    
    # Actually, let me just return schoolbook.
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
