"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s
    
    # Schoolbook decomposition: rank = n*m*p = 52
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try rank 41 = 3p+2 construction using dual numbers / polynomial embedding
    target_rank = 3 * p + 2  # 41
    if target_rank < best_rank and time.time() < deadline - 5:
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Construction: Use the fact that for 2x2 x 2xp, we can use
        # a "chain" of shared multiplications.
        #
        # Key insight: For adjacent columns k and k+1, we can share
        # intermediate products. Define:
        #   For each k in 0..p-1:
        #     a_k = A[0,0]  (same for all k)
        #     b_k = A[0,1]  (same for all k)  
        #     c_k = A[1,0]  (same for all k)
        #     d_k = A[1,1]  (same for all k)
        #
        # We use the following 3-multiplication pattern per "step":
        #   m1 = (A[0,0] + A[1,0]) * B[0,k]
        #   m2 = A[1,1] * B[1,k]
        #   m3 = (A[0,1] - A[1,0]) * (B[0,k] + B[1,k])
        #
        # Then:
        #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        #            = (m1 - A[1,0]*B[0,k]) + (m3 - A[0,1]*B[0,k] + A[1,0]*B[1,k] - A[1,0]*B[0,k])
        # This is getting messy. Let me use a cleaner approach.
        #
        # CLEANER CONSTRUCTION:
        # For the 2x2 x 2xp tensor, use the following rank 3p+2 decomposition:
        #
        # Terms 0, 1: Left boundary (k=0)
        #   Term 0: U[0,0]=1, V[0,0]=1, W[0,0]=1  -> A[0,0]*B[0,0] -> C[0,0]
        #   Term 1: U[1,1]=1, V[1,0]=1, W[0,1]=1  -> A[0,1]*B[1,0] -> C[0,0]
        #
        # Wait, I need to be more careful. Let me use a verified construction.
        #
        # VERIFIED CONSTRUCTION for M_{2,2,p} with rank 3p+2:
        #
        # For s = 0:
        #   r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1
        #   r=1: U[1,1]=1, V[1,0]=1, W[0,1]=1
        #
        # For s = 1 to p-1 (interior):
        #   r=2+3*(s-1):   U[0,0]=1, U[1,0]=1, V[0,s]=1, W[s,0]=1, W[s,1]=1
        #   r=2+3*(s-1)+1: U[1,1]=1, V[1,s]=1, W[s,0]=1, W[s,1]=1
        #   r=2+3*(s-1)+2: U[0,1]=1, U[1,0]=1, V[0,s]=1, V[1,s]=1, W[s,0]=1, W[s,1]=1
        #
        # For s = p-1 (right boundary):
        #   r=2+3*(p-2): U[0,0]=1, V[0,p-1]=1, W[p-1,0]=1
        #   r=2+3*(p-2)+1: U[1,1]=1, V[1,p-1]=1, W[p-1,1]=1
        #
        # Total: 2 + 3*(p-1) + 2 = 3p+1. Not quite 3p+2.
        #
        # Let me just implement the schoolbook and verify it works, then try to
        # find a better construction through direct algebraic manipulation.
        
        # Working construction for rank 3p+2:
        # 
        # Use the following pattern. For each k from 0 to p-1, we need:
        #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        #   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        #
        # We use 3 multiplications per "block" with overlap:
        #   For block k (0 <= k <= p-2):
        #     m_{3k}   = (A[0,0] + A[1,0]) * (B[0,k] + B[0,k+1])
        #     m_{3k+1} = (A[0,1] + A[1,1]) * (B[1,k] + B[1,k+1])
        #     m_{3k+2} = (A[0,0] + A[0,1] + A[1,0] + A[1,1]) * (B[0,k] + B[1,k] + B[0,k+1] + B[1,k+1])
        #
        # This is getting too complex. Let me use a simpler verified approach:
        #
        # DIRECT CONSTRUCTION:
        # Rank 3p+2 decomposition:
        #
        # Terms 0, 1: handle column 0
        #   t0: A[0,0]*B[0,0] -> contributes to C[0,0]
        #   t1: A[1,1]*B[1,0] -> contributes to C[1,0]
        #
        # Terms 2 to 2+3*(p-2)-1: for k=1 to p-2, use 3 terms each
        #   For column k:
        #     t_{3(k-1)+2}: (A[0,1]+A[1,0])*(B[0,k]+B[1,k])
        #     t_{3(k-1)+3}: A[0,0]*B[0,k]
        #     t_{3(k-1)+4}: A[1,1]*B[1,k]
        #
        # Terms 2+3*(p-2) to end: handle column p-1
        #   t_{2+3*(p-2)}: A[0,0]*B[0,p-1]
        #   t_{2+3*(p-2)+1}: A[1,1]*B[1,p-1]
        #
        # This gives 2 + 3*(p-2) + 2 = 3p-2 terms. Not enough.
        #
        # Block Strassen approach: group columns into pairs
        # For 2x2 x 2x2: Strassen rank 7
        # For 2x2 x 2x13: 6 pairs (rank 7 each) + 1 single (rank 4) = 6*7 + 4 = 46
        
        # Strassen decomposition for 2x2 x 2x2:
        # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # m2 = (A[1,0]+A[1,1]) * B[0,0]
        # m3 = A[0,0] * (B[1,1]-B[0,0])
        # m4 = A[1,1] * (B[0,0]-B[0,1])  # Wait, this is for 2x2x2x2
        #
        # For 2x2 x 2x2 (A is 2x2, B is 2x2, C is 2x2):
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        #
        # Strassen (7 multiplications):
        # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # m2 = (A[1,0]+A[1,1]) * B[0,0]
        # m3 = A[0,0] * (B[1,1]-B[0,0])
        # m4 = A[1,1] * (B[0,0]-B[0,1])  # This is wrong
        #
        # Correct Strassen for matrix multiplication M_{2,2,2}:
        # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # m2 = (A[1,0]+A[1,1]) * B[0,0]
        # m3 = A[0,0] * (B[1,1]-B[0,0])
        # m4 = A[1,1] * (B[0,0]-B[0,1])  # Still wrong
        #
        # Let me look this up properly. Strassen for 2x2x2:
        # m1 = (A11 + A22)(B11 + B22)
        # m2 = (A21 + A22) B11
        # m3 = A11 (B21 - B11)
        # m4 = A22 (B12 - B22)
        # m5 = (A11 + A12) B22
        # m6 = (A21 - A11)(B11 + B12)
        # m7 = (A12 - A22)(B21 + B22)
        #
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        #
        # So for 2x2 x 2x2, rank 7 works!
        
        # Now for 2x2 x 2x13:
        # Group columns: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), and 12
        # Each pair uses 7 rank terms, single uses 4 rank terms
        # Total: 6*7 + 4 = 46
        
        rank_block = 6 * 7 + 4  # 46
        if rank_block < best_rank and time.time() < deadline - 5:
            U = [[0] * rank_block for _ in range(n * m)]
            V = [[0] * rank_block for _ in range(m * p)]
            W = [[0] * rank_block for _ in range(p * n)]
            
            # Strassen coefficients for 2x2 x 2x2
            # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
            # m2 = (A[1,0]+A[1,1]) * B[0,0]
            # m3 = A[0,0] * (B[1,1]-B[0,0])
            # m4 = A[1,1] * (B[0,0]-B[0,1])  # WRONG - let me use correct formula
            #
            # Correct Strassen:
            # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
            # m2 = (A[1,0]+A[1,1]) * B[0,0]
            # m3 = A[0,0] * (B[1,1]-B[0,0])
            # m4 = A[1,1] * (B[0,0]-B[0,1])  # Still wrong
            #
            # I'll use the standard Strassen formula:
            # m1 = (A11+A22)(B11+B22)
            # m2 = (A21+A22)B11
            # m3 = A11(B21-B11)
            # m4 = A22(B12-B22)
            # m5 = (A11+A12)B22
            # m6 = (A21-A11)(B11+B12)
            # m7 = (A12-A22)(B21+B22)
            #
            # C11 = m1+m4-m5+m7
            # C12 = m3+m5
            # C21 = m2+m4
            # C22 = m1-m2+m3+m6
            
            # For our case, A is 2x2, B is 2x2 (two columns), C is 2x2
            # B[0,0], B[0,1] are row 0 of B, B[1,0], B[1,1] are row 1 of B
            
            # Strassen terms for a pair of columns (k1, k2):
            # m1: U[0]=1, U[3]=1, V[k1]=1, V[2+k2]=1, W[k1*2]=1, W[k2*2]=1
            # m2: U[2]=1, U[3]=1, V[k1]=1, W[k1*2+1]=1
            # m3: U[0]=1, V[2+k2]=1, V[k1]=-1, W[k2*2+1]=1
            # m4: U[3]=1, V[k1]=1, V[2+k2]=-1, W[k1*2+1]=1
            # m5: U[0]=1, U[1]=1, V[2+k2]=1, W[k2*2]=1
            # m6: U[2]=1, U[0]=-1, V[k1]=1, V[2+k1]=1, W[k2*2+1]=1
            # m7: U[1]=1, U[3]=-1, V[2+k1]=1, V[2+k2]=1, W[k1*2]=1
            
            # Let me verify m1: (A[0,0]+A[1,1])*(B[0,k1]+B[1,k2])
            # U: A[0,0] and A[1,1] -> indices 0 and 3
            # V: B[0,k1] and B[1,k2] -> indices k1 and 2+k2
            # W: contributes to C[0,k1] and C[1,k2] -> indices k1*2+0 and k2*2+1
            # But this doesn't give the right C entries...
            
            # I think I'm overcomplicating this. Let me just return schoolbook.
            pass
        
    return best_result
# EVOLVE-BLOCK-END
