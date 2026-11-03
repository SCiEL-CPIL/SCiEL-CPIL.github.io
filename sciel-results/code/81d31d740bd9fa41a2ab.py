"""Program for rank-minimal tensor decomposition of 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    deadline = time.time() + 115  # safety margin before 120s limit

    # The matrix multiplication tensor T for (n, m, p) = (2, 3, 6)
    # has entries T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The decomposition requires: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy: Try to find a rank-18 decomposition using the structure of the problem.
    # The key insight is that M_{2,3,6} can be decomposed more efficiently than schoolbook (36).
    
    # Let's try a direct construction with rank 18.
    # We'll use the fact that we can share computations between columns.
    
    rank = 18
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Construction approach:
    # For each j in {0,1,2} and k in {0,1,2,3,4,5}, we need C[i,k] += A[i,j]*B[j,k]
    # 
    # We can group the 6 columns into 3 pairs: (0,3), (1,4), (2,5)
    # For each pair (k1, k2) and each j, we can use 2 multiplications instead of 4
    # by using identities like:
    #   (a1+b1)*(b1+b2) = a1*b1 + a1*b2 + b1*b1 + b1*b2
    #   This doesn't directly help, but we can use:
    #   (a1+a2)*(b1+b2) - a1*b1 - a2*b2 = a1*b2 + a2*b1
    #
    # Actually, let me try a different approach. Let's use the known result that
    # rank(M_{2,3,3}) = 15 and see if we can extend it.
    #
    # For M_{2,3,3}, a rank-15 decomposition exists. Let me try to construct one.
    #
    # Actually, let me try a simpler approach: use rank 24 by splitting into two 3-column blocks
    # and using a rank-12 decomposition for each 2x3 by 3x3 block.
    #
    # For 2x3 by 3x3, schoolbook is 9. Can we do better?
    # The rank of M_{2,3,3} is known to be 15 (not 9 or 12).
    #
    # Let me just try to construct a valid decomposition with rank < 36.
    # I'll use a systematic approach: try rank 24 first (2 * 12 for two 3-column blocks).
    
    # Actually, let me try rank 18 directly with a specific construction.
    # 
    # Key idea: For each j (row of B / column of A), we have 6 B entries.
    # We can use 3 rank-1 terms per j to cover all 6 k values for both i=0 and i=1.
    #
    # For j=0:
    #   Term 0: U[0,0]=1, V[0,0]=1, W[0,0]=1  -> A[0,0]*B[0,0] -> C[0,0]
    #   Term 1: U[0,1]=1, V[1,1]=1, W[1,1]=1  -> A[0,0]*B[0,1] -> C[0,1]
    #   ...
    #
    # This is getting complex. Let me just return the correct schoolbook decomposition
    # with rank 36, which is guaranteed to be valid.
    
    # Actually, let me try one more thing: a rank-24 construction.
    # Split B into B_left (3x3) and B_right (3x3).
    # For each half, use a rank-12 decomposition for 2x3 by 3x3.
    # 
    # For 2x3 by 3x3, here's a rank-12 construction (not optimal but valid):
    # We need 12 rank-1 terms. Schoolbook is 9, so rank-12 is worse.
    # Let me just use schoolbook for each half: rank 9 each, total 18.
    # But that's the same as schoolbook for the whole thing (36/2=18 per half, total 36).
    #
    # OK, I'll just return the correct rank-36 schoolbook decomposition.
    
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
