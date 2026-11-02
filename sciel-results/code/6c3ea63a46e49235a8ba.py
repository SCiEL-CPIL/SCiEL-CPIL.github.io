"""Program for rank-minimal 2x3x4 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=4
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (12, rank)
    # W: (p*n, rank) = (8, rank)
    
    # The tensor is T_{2,3,4}.
    # We can split p=4 into two blocks of size 2.
    # Block 1: columns 0,1 of B.
    # Block 2: columns 2,3 of B.
    # Each block corresponds to multiplying A (2x3) by B_block (3x2) to get C_block (2x2).
    # The tensor for 2x3x2 is isomorphic to 2x2x3.
    # The rank of 2x2x3 is 11.
    # So the rank of 2x3x4 is at most 11 + 11 = 22.
    
    # We need a rank-11 decomposition for 2x3x2.
    # Let's find a rank-11 decomposition for 2x3x2.
    # A (2x3), B (3x2), C (2x2).
    # U: (6, 11), V: (6, 11), W: (4, 11).
    # Indices:
    # U: i*3+j for i in {0,1}, j in {0,1,2}
    # V: j*2+k for j in {0,1,2}, k in {0,1}
    # W: k*2+i for k in {0,1}, i in {0,1}
    
    # Let's use the known rank-11 decomposition for 2x2x3 and map it.
    # Or derive it directly.
    # Actually, 2x3x2 is isomorphic to 2x2x3.
    # A (2x3) * B (3x2) = C (2x2).
    # Let A' = A^T (3x2), B' = B^T (2x3).
    # C' = C^T = B' A' (2x2).
    # This is not exactly 2x2x3.
    # 2x2x3: A (2x2) * B (2x3) = C (2x3).
    
    # Let's just construct a rank-22 decomposition directly.
    # We will use the schoolbook decomposition for each 2x3x2 block.
    # Schoolbook rank for 2x3x2 is 2*3*2 = 12.
    # So total rank would be 24.
    # We need to get 22.
    
    # Let's try to find a rank-11 decomposition for 2x3x2.
    # This is equivalent to finding a rank-11 decomposition for 2x2x3.
    # Let's look up the rank-11 factors for 2x2x3.
    # From literature, a rank-11 decomposition for 2x2x3 is known.
    # Let's use the following factors for 2x2x3 (n=2, m=2, p=3):
    # U (4x11), V (6x11), W (6x11).
    
    # I will use a known set of factors.
    # U = [
    # [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    # [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # ]
    # V = [
    # [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    # [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    # [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
    # [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
    # [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    # ]
    # W = [
    # [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    # ]
    
    # Let's verify this decomposition for 2x2x3.
    # T_{i,j,k} = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # i in {0,1}, j in {0,1}, k in {0,1,2}
    # U indices: i*2+j
    # V indices: j*3+k
    # W indices: k*2+i
    
    # c=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> T[0,0,0] += 1
    # c=1: U[0,1]=1, V[1,0]=1, W[0,1]=1 -> T[0,1,0] += 1
    # c=2: U[1,0]=1, V[0,1]=1, W[1,0]=1 -> T[1,0,1] += 1
    # c=3: U[0,0]=1, U[1,1]=0, V[1,0]=1, V[0,1]=0, W[0,0]=1, W[1,1]=1
    # Wait, U[1,1] is row 3, col 3. U[3][3]=0.
    # U[0][3]=1. So U[0,0]=1.
    # V[1][3]=1. So V[1,0]=1.
    # W[0][3]=0. W[1][3]=1.
    # T[0,0,0] += U[0,0,3] V[0,0,3] W[0,0,3] = 1*0*0 = 0
    # T[0,1,0] += U[0,1,3] V[1,0,3] W[0,0,3] = 0*1*0 = 0
    # T[1,0,0] += U[1,0,3] V[0,0,3] W[0,1,3] = 0*0*0 = 0
    # T[1,1,0] += U[1,1,3] V[1,0,3] W[0,1,3] = 0*1*0 = 0
    # T[0,0,1] += U[0,0,3] V[0,1,3] W[1,0,3] = 1*0*0 = 0
    # T[0,1,1] += U[0,1,3] V[1,1,3] W[1,0,3] = 0*0*0 = 0
    # T[1,0,1] += U[1,0,3] V[0,1,3] W[1,1,3] = 0*0*1 = 0
    # T[1,1,1] += U[1,1,3] V[1,1,3] W[1,1,3] = 0*0*1 = 0
    # T[0,0,2] += U[0,0,3] V[0,2,3] W[2,0,3] = 1*0*0 = 0
    # T[0,1,2] += U[0,1,3] V[1,2,3] W[2,0,3] = 0*0*0 = 0
    # T[1,0,2] += U[1,0,3] V[0,2,3] W[2,1,3] = 0*0*0 = 0
    # T[1,1,2] += U[1,1,3] V[1,2,3] W[2,1,3] = 0*0*0 = 0
    
    # This factor c=3 seems to contribute nothing?
    # U[3][3]=0, U[0][3]=1.
    # V[1][3]=1, V[3][3]=1.
    # W[1][3]=1, W[3][3]=1.
    # T[i,j,k] += U[i*2+j, 3] V[j*3+k, 3] W[k*2+i, 3]
    # If i=0, j=0, k=0: U[0,3]=1, V[0,3]=0, W[0,3]=0. 0.
    # If i=0, j=1, k=0: U[1,3]=0, V[3,3]=1, W[0,3]=0. 0.
    # If i=1, j=0, k=0: U[2,3]=0, V[0,3]=0, W[1,3]=1. 0.
    # If i=1, j=1, k=0: U[3,3]=0, V[3,3]=1, W[1,3]=1. 0.
    # If i=0, j=0, k=1: U[0,3]=1, V[1,3]=1, W[2,3]=0. 0.
    # If i=0, j=1, k=1: U[1,3]=0, V[4,3]=0, W[2,3]=0. 0.
    # If i=1, j=0, k=1: U[2,3]=0, V[1,3]=1, W[3,3]=1. 0.
    # If i=1, j=1, k=1: U[3,3]=0, V[4,3]=0, W[3,3]=1. 0.
    # If i=0, j=0, k=2: U[0,3]=1, V[2,3]=0, W[4,3]=0. 0.
    # If i=0, j=1, k=2: U[1,3]=0, V[5,3]=0, W[4,3]=0. 0.
    # If i=1, j=0, k=2: U[2,3]=0, V[2,3]=0, W[5,3]=0. 0.
    # If i=1, j=1, k=2: U[3,3]=0, V[5,3]=0, W[5,3]=0. 0.
    
    # So c=3 contributes nothing. That's bad.
    # The factors I found are likely incorrect.
    
    # I will use the schoolbook rank 24 to ensure correctness.
    # The task is to improve, but correctness is paramount.
    # If I can't find a valid rank-22, I'll stick with 24.
    
    # Let's try to construct a rank-22 decomposition by combining two rank-11 blocks.
    # But I need a valid rank-11 for 2x3x2.
    # Let's use the rank-12 schoolbook for each block, total 24.
    # Then try to merge two factors to get 22?
    # Merging two factors in 2x3x2:
    # Factor 1: A[0,0]B[0,0]C[0,0]
    # Factor 2: A[0,1]B[1,0]C[1,0]
    # These are independent.
    
    # I will return the schoolbook rank 24.
    
    rank = 24
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
