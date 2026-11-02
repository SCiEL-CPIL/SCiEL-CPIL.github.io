import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    
    # We want to decompose the matrix multiplication tensor T_{n,m,p} of shape (n*m, m*p, p*n)
    # The index convention is:
    #   U has shape (n*m, rank), indexed by (i, j) where i in [0,n), j in [0,m)
    #   V has shape (m*p, rank), indexed by (j, k) where j in [0,m), k in [0,p)
    #   W has shape (p*n, rank), indexed by (k, i) where k in [0,p), i in [0,n)
    #   The decomposition satisfies: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # The directive suggests a "slice-and-dice" approach:
    # Partition the 3x4 matrix into blocks:
    #   Rows: 3 = 2 + 1
    #   Cols: 4 = 2 + 2
    # This gives four sub-problems:
    #   (2x2) * (2x7) -> 2x2x7
    #   (2x2) * (2x7) -> 2x2x7
    #   (1x2) * (2x7) -> 1x2x7
    #   (1x2) * (2x7) -> 1x2x7
    #
    # The directive claims rank(2x2x7) = 14 and rank(1x2x7) = 2, giving total rank = 2*14 + 2*2 = 32.
    # However, rank(1x2x7) = 1*2*7 = 14 (since for 1xm x m x p, the rank is m*p).
    # And rank(2x2x7): the schoolbook is 2*2*7 = 28. Can we do better?
    # For 2x2 matrix multiplication, Strassen gives rank 7 for 2x2x2.
    # For 2x2x7, we can potentially use a Strassen-like approach.
    # Actually, the rank of 2x2x7 tensor is at most 2*2*7 = 28 (schoolbook).
    # Using Strassen for the 2x2 part: we need to multiply a 2x2 matrix by a 2x7 matrix.
    # This is equivalent to 7 independent 2x2 matrix multiplications (one for each column of the 2x7).
    # Each 2x2x2 has rank 7 (Strassen), so 2x2x7 has rank 7*7 = 49? That's worse.
    # Wait, no. The tensor for 2x2x7 is T[i,j,k] where i in {0,1}, j in {0,1}, k in {0,...,6}.
    # This is a sum over j: sum_j A[i,j] * B[j,k].
    # For each k, we're doing a 2x2 matrix-vector multiplication, which is a 2x2x1 tensor.
    # The 2x2x1 tensor has rank 2 (since it's just two independent linear combinations).
    # So 2x2x7 has rank at most 7*2 = 14? Let's check.
    # Actually, T_{2,2,7} = sum_{k=0}^{6} T_{2,2,1}^{(k)}, where each T_{2,2,1}^{(k)} has rank 2.
    # But these are in different slices, so the total rank is 7*2 = 14.
    # Yes! For 2x2x7, we can do rank 14 by treating each output column independently.
    # For 1x2x7, similarly, each output column is a 1x2x1 tensor with rank 1, so total rank 7*1 = 7.
    # Wait, 1x2x1: T[i,j,k] with i=0, j in {0,1}, k=0. T[0,j,0] = A[0,j] * B[j,0].
    # This is rank 1? No, it's sum_j A[0,j] * B[j,0], which is a single product, so rank 1.
    # So 1x2x7 has rank 7.
    # Total rank: 2 * 14 + 2 * 7 = 28 + 14 = 42.
    
    # Let's implement this.
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 42
    V = [[0] * rank for _ in range(m * p)]  # 28 x 42
    W = [[0] * rank for _ in range(p * n)]  # 21 x 42
    
    col = 0
    # Block 1: rows 0-1, cols 0-1 (2x2 x 7)
    # This handles A[0:2, 0:2] * B[0:2, 0:7]
    for k in range(p):
        for t in range(2):  # t=0,1 for the two rank-1 terms in 2x2x1
            r = col + t
            if t == 0:
                # Term 1: A[0,0]*B[0,k] + A[0,1]*B[1,k] for output (0,k)
                # And A[1,0]*B[0,k] for output (1,k)
                # This isn't quite right. Let me think more carefully.
                pass
            col += 1
    
    # Let me reconsider. For 2x2x1:
    # Output C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # Output C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # This is a tensor of rank 2:
    #   Term 1: U[0,0]=1, U[1,0]=0, V[0,0]=1, V[1,0]=0, W[0,0]=1, W[1,0]=0 -> A[0,0]*B[0,0]*C[0,0]
    #   Term 2: U[0,1]=1, U[1,1]=1, V[0,1]=0, V[1,1]=1, W[0,1]=0, W[1,1]=1 -> A[0,1]*B[1,0]*C[0,0] + A[1,1]*B[1,0]*C[1,0]
    #   But we also need A[1,0]*B[0,0]*C[1,0]
    # So rank 2 isn't enough for 2x2x1. The rank of 2x2x1 is actually 4 (schoolbook).
    
    # I was wrong. Let me go back to schoolbook for each sub-problem.
    # 2x2x7: rank 28
    # 1x2x7: rank 14
    # Total: 2*28 + 2*14 = 84. Same as before.
    
    # Let me just use the schoolbook decomposition.
    rank = n * m * p  # 84
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
                
    return {"U": U, "V": V, "W": W}
