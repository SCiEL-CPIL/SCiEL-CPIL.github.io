"""Program for matrix multiplication 2x5 by 5x6 using rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # For M_{2,5,6}, the rank is known to be 54 (less than schoolbook 60).
    # We'll use a constructive approach based on the known result.
    
    # The key insight: for M_{2,m,p}, the rank can be reduced by using
    # shared intermediate computations across the two output rows.
    
    # A standard construction for M_{2,5,6} with rank 54:
    # Use the fact that R(M_{2,m,p}) <= 2*max(m,p) - 1 when min(m,p) >= 2
    # More precisely, for M_{2,m,p}, rank = 2*max(m,p) - 1 + (m-1)*(p-1) ... 
    # Actually, let me use a direct construction.
    
    # For 2x5x6: rank is 54.
    # Construction:
    # For each column j of A (5 columns) and row k of B (6 rows), we need
    # products A[i,j]*B[j,k] for i=0,1.
    
    # The standard schoolbook uses 2*5*6 = 60 terms.
    # We can reduce this by sharing computations.
    
    # One known construction for M_{2,m,p}:
    # rank = m*p + (m-1)*(p-1) = 30 + 4*5 = 50? No, that's not right either.
    
    # Let me use the actual known result. For M_{2,5,6}, the tensor rank is 54.
    
    # A constructive approach:
    # For columns j=0 to 4 of A:
    #   For rows k=0 to 5 of B:
    #     We need A[0,j]*B[j,k] and A[1,j]*B[j,k]
    #     
    # Use the following scheme:
    # For each j, define:
    #   S_j = A[0,j] + A[1,j]
    #   D_j = A[0,j] - A[1,j]
    # Then:
    #   A[0,j] = (S_j + D_j)/2
    #   A[1,j] = (S_j - D_j)/2
    #
    # So A[0,j]*B[j,k] = (S_j + D_j)*B[j,k]/2
    #    A[1,j]*B[j,k] = (S_j - D_j)*B[j,k]/2
    #
    # This gives for each (j,k): 2 terms (S_j*B[j,k] and D_j*B[j,k])
    # Total: 2 * 5 * 6 = 60 terms. Same as schoolbook!
    
    # The reduction comes from a different strategy. Let me think about this differently.
    
    # Actually, the known rank for M_{2,5,6} is indeed 54, which can be achieved
    # by a more sophisticated construction. Let me implement the known optimal
    # decomposition.
    
    # For now, use schoolbook (rank 60) which is guaranteed valid.
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
# EVOLVE-BLOCK-END
