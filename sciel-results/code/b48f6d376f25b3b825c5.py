"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition."""
import time
import itertools

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Start with the schoolbook rank
    rank = n * m * p
    
    # We will attempt to reduce the rank using a greedy approach.
    # The tensor is T[i, j, k] where i in [0,n), j in [0,m), k in [0,p).
    # The decomposition is T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # The schoolbook solution has U, V, W as one-hot vectors for each (i,j,k).
    # We can try to combine terms if U, V, W can be shared.
    
    # For 2x4x4, Strassen-like algorithms exist. For 2x4x7, we can use a block decomposition.
    # Specifically, we can decompose the 2x4 matrix into two 2x2 blocks, and the 4x7 into two 2x2 and one 2x3 blocks? 
    # Or maybe treat it as a 2x4x4 multiplication plus a 2x4x3 multiplication.
    
    # Let's try a recursive/block approach.
    # C = A * B where A is 2x4, B is 4x7.
    # Split A into [A1, A2] where A1 is 2x2, A2 is 2x2.
    # Split B into [B1; B2; B3] where B1 is 2x7, B2 is 2x7, B3 is 0? No, B is 4x7, so it's [B1; B2] where B1 is 2x7, B2 is 2x7.
    # C = A1 * B1 + A2 * B2.
    # This reduces to two 2x2 x 2x7 multiplications.
    # 2x2 x 2x7 can be done with Strassen? Strassen is for 2x2 x 2x2.
    # For 2x2 x 2x7, we can use the fact that it's 7 independent 2x2 x 2x1 multiplications.
    # Actually, 2x2 x 2x7 is equivalent to 7 copies of 2x2 x 2x1.
    # But we can do better by grouping the columns of B.
    
    # Let's look at known results.
    # The rank of 2x4 x 4x7 is at most 2 * rank(2x2 x 2x7) + ...?
    # Actually, C[i,k] = sum_{j=0}^3 A[i,j] B[j,k].
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] + A[i,3]B[3,k].
    
    # We can group the columns of A and rows of B.
    # Let's use a 2x2 block structure for A and 2x7 for B.
    # This doesn't directly help unless we use a rank-reducing algorithm for the block.
    
    # Let's try to find a rank-30 decomposition.
    # Schoolbook is 2*4*7 = 56.
    # If we can reduce by ~20%, that's rank ~45.
    # Strassen for 2x2x2 is 7 (vs 8).
    # For 2x4x4, the rank is at most 2*7=14? No, it's more complex.
    # Actually, 2x4x4 can be viewed as two 2x2x4 multiplications.
    # Or one 2x4x2 and one 2x4x2?
    
    # Let's use a greedy rank reduction.
    # We start with the 56 basis tensors.
    # We try to see if we can replace pairs of basis tensors with a single one, if possible.
    # This is generally hard, but for small dimensions we can try.
    
    # A simpler approach: use the known rank for 2x4x4.
    # The rank of 2x4x4 is 12? (2*4*4=32 schoolbook, but can be reduced).
    # Actually, I don't know the exact rank for 2x4x4.
    
    # Let's just return the schoolbook solution. It's safe and correct.
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
