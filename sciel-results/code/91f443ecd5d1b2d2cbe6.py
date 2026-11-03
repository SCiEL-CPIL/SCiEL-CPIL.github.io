import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # We'll use Strassen on 2x2 blocks where possible, and schoolbook for the rest.
    # A is 3x3, B is 3x8, C is 3x8.
    # We can split A into a 2x2 block (top-left) and the remaining rows/cols.
    # Actually, let's think about this differently.
    # The known rank for 3x3x3 is 22.
    # For 3x3x8, we can think of it as 8 columns of 3x3x1.
    # Rank(3x3x1) = 9 (schoolbook).
    # But we can do better by using Strassen on the 3x3 part.
    # 
    # Let's use the decomposition from the literature.
    # For 3x3x3, the minimum rank is 22 (de Groote 1979, then improved to 23? No, 22 is known).
    # Actually, the minimum rank for 3x3x3 is 22.
    # For 3x3x8, we can use 8 copies of 3x3x1 with rank 9 each = 72 (schoolbook).
    # But we can do better by using the 22-rank decomposition for 3x3x3 and then extending.
    #
    # Actually, a better approach: use Strassen's algorithm on 2x2 blocks.
    # Split A into:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # Split B into:
    # B = [[B11, B12, B13, B14], [B21, B22, B23, B24], [B31, B32, B33, B34]]
    # where B11 is 2x4, etc.
    #
    # This is getting complex. Let me just implement the schoolbook decomposition first,
    # which gives rank 72, and then try to improve.
    #
    # Actually, let's use a known result: the rank of 3x3x8 is at most 62.
    # Here's a construction:
    # Use Strassen on the 2x2 top-left block of A, and schoolbook for the rest.
    #
    # Let's define the indices:
    # A[i,j] for i,j in 0..2
    # B[j,k] for j in 0..2, k in 0..7
    # C[i,k] for i in 0..2, k in 0..7
    #
    # U has shape (9, rank), V has shape (24, rank), W has shape (24, rank).
    # U index: i*m+j = 3*i+j
    # V index: j*p+k = 8*j+k
    # W index: k*n+i = 8*k+i
    #
    # Let's use the following strategy:
    # 1. For the 2x2 top-left block of A (A[0:2,0:2]), use Strassen with B[0:2,0:8].
    #    This gives 7 multiplications instead of 4*8=32.
    #    Wait, Strassen is for 2x2x2. For 2x2x8, we can apply Strassen column-wise.
    #    Actually, for 2x2x8, the rank is 7*8/2? No.
    #    For 2x2xp, the rank is 7p/2 if p is even? No, it's 7p for p=2, but for general p,
    #    it's 7p if we do it column by column, but we can do better.
    #    Actually, rank(2x2xp) = 7p for p>=2? No, that's not right.
    #    For 2x2x2, rank=7. For 2x2x4, rank=14? Or can we do better?
    #    I think rank(2x2xp) = 7p for p>=2 is an upper bound, but the exact minimum is unknown for large p.
    #
    # Let's just do schoolbook for everything first to get a valid solution, then optimize.
    
    # Schoolbook: rank = 3*3*8 = 72
    rank = 72
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Verify the decomposition
    # C = sum_r U[:,r] * V[:,r]^T * W[:,r]
    # Let's check with random matrices
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        C_computed += u @ v @ w
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to pure schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb))
        V_fb = np.zeros((m*p, rank_fb))
        W_fb = np.zeros((p*n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
