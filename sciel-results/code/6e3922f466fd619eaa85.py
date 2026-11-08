import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    def naive_solution():
        rank = n * m * p  # 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, rank):
        np.random.seed(42)
        X = np.random.randn(n, m)
        Y = np.random.randn(m, p)
        Z_expected = X @ Y
        Z_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        Z_computed[i, k] += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
        return np.allclose(Z_expected, Z_computed, atol=1e-10)
    
    # Try to find a rank-48 solution using a structured approach.
    # Key insight: Z[i,k] = sum_j X[i,j] * Y[j,k]
    # For fixed i, this is a dot product of row i of X with column k of Y.
    # We can precompute linear combinations of X's rows.
    
    # Approach: Use a rank-6 scheme for computing 2 dot products of length 4.
    # For each output column k, we need:
    #   Z[0,k] = X[0,0]*Y[0,k] + X[0,1]*Y[1,k] + X[0,2]*Y[2,k] + X[0,3]*Y[3,k]
    #   Z[1,k] = X[1,0]*Y[0,k] + X[1,1]*Y[1,k] + X[1,2]*Y[2,k] + X[1,3]*Y[3,k]
    
    # Let's try to find 6 linear combinations S_r of the 8 X-entries such that
    # we can recover both Z[0,k] and Z[1,k] from S_r and Y[:,k].
    
    # This is equivalent to finding a rank-6 decomposition of the 2x4x4x1 tensor.
    # Let me try to construct one.
    
    # For the 2x4 x 4x1 problem, the tensor T[i,j] = 1 (i in {0,1}, j in {0,1,2,3})
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    # Wait, that's not quite right. Let me re-derive.
    
    # Actually for 2x4 x 4x1: Z[i] = sum_j X[i,j] * Y[j]
    # Tensor: T[i,j] = 1 for all i,j (since each X[i,j]*Y[j] contributes to Z[i])
    # Decomposition: T[i,j] = sum_r U[i*m+j, r] * V[j*1+0, r] * W[0*n+i, r]
    # But V has shape (m*p, rank) = (4*1, rank) = (4, rank) and W has shape (p*n, rank) = (1*2, rank) = (2, rank)
    
    # So: for all i in {0,1}, j in {0,1,2,3}:
    #   sum_r U[i*4+j, r] * V[j, r] * W[i, r] = 1
    
    # This is a system of 8 equations. Let's try rank 6.
    # Variables: 8*6 + 4*6 + 2*6 = 48+24+12 = 84 unknowns, 8 equations. Very underdetermined.
    
    # Let me try a simple construction:
    # Use the identity: 1 = 1*1, so for each (i,j), set one term to 1.
    # But that's rank 8. Can we do better?
    
    # Try: for j=0,1: use U[i*4+j, r] = 1, V[j, r] = 1, W[i, r] = 1/2 for r=0,1
    #       for j=2,3: use U[i*4+j, r] = 1, V[j, r] = 1, W[i, r] = 1/2 for r=2,3
    # This gives rank 4 but doesn't work because W[i,r] would need to be 1 for all r.
    
    # Let me think differently. The constraint is:
    # sum_r U[i*4+j, r] * V[j, r] * W[i, r] = 1 for all i,j
    
    # If we set W[i, r] = 1 for all i,r, then we need:
    # sum_r U[i*4+j, r] * V[j, r] = 1 for all i,j
    # This is a rank-1 condition on the 2x4 matrix M[i,j] = sum_r U[i*4+j,r]*V[j,r]
    # But M[i,j] must equal 1 for all i,j, which is rank 1 (all ones).
    # So we can set U[i*4+j, r] = 1, V[j, r] = 1 for r=0, and 0 otherwise.
    # But that's rank 1! Wait, that can't be right.
    
    # Oh wait, I'm confusing myself. The decomposition is for the tensor, not for a matrix.
    # Let me re-read the problem.
    
    # The tensor for (n x m) * (m x p) is T[i,j,k] where:
    # - i indexes rows of X (0..n-1)
    # - j indexes columns of X / rows of Y (0..m-1)  
    # - k indexes columns of Y (0..p-1)
    # T[i,j,k] = 1 for all valid (i,j,k)
    
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For n=2, m=4, p=8:
    # T[i,j,k] = sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r] = 1 for all i,j,k
    
    # This is 2*4*8 = 64 equations.
    
    # Let me try to find a rank-48 solution by exploiting the structure.
    # Key: for fixed (i,k), sum_j [U[i*4+j,r] * V[j*8+k,r] * W[k*2+i,r]] = 1
    # But actually it's not sum_j, it's that for each (i,j,k) individually, the sum over r equals 1.
    
    # So we need: for all i in {0,1}, j in {0,1,2,3}, k in {0,...,7}:
    #   sum_{r=0}^{R-1} U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r] = 1
    
    # This is a very underdetermined system for R >= 8.
    
    # Let me try a constructive approach with rank 48.
    # Idea: group the 8 k-values into 4 pairs, and use a rank-12 scheme per pair.
    # Or: use the fact that X has only 2 rows to precompute combinations.
    
    # Let me try a different structural approach:
    # Precompute 6 linear combinations of each row of X.
    # S[r, i, j] = sum over some pattern... 
    
    # Actually, let me just try to find ANY valid solution with rank < 56 by solving the system.
    # I'll use a simple iterative approach.
    
    # For now, let me try the rank-56 Strassen solution and verify it works.
    
    # Strassen 2x2x2 decomposition (verified)
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, 0, 1],
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 1],
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
    ]
    
    # Verify Strassen works
    A = np.random.randn(2, 2)
    B = np.random.randn(2, 2)
    C_expected = A @ B
    C_computed = np.zeros((2, 2))
    for r in range(7):
        a_r = sum(strassen_U[r][idx] * A[idx // 2, idx % 2] for idx in range(4))
        b_r = sum(strassen_V[r][idx] * B[idx // 2, idx % 2] for idx in range(4))
        for idx in range(4):
            i, k = idx // 2, idx % 2
            C_computed[i, k] += a_r * b_r * strassen_W[r][idx]
    strassen_valid = np.allclose(C_expected, C_computed, atol=1e-10)
    
    if not strassen_valid:
        return naive_solution()
    
    # Build rank-56 solution using Strassen blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(base_rank, x_block, y_block_row, y_block_col):
        for r in range(7):
            cr = base_rank + r
            for idx in range(4):
                li, lj = divmod(idx, 2)
                U[li * m + (x_block * 2 + lj)][cr] += strassen_U[r][idx]
            for idx in range(4):
                lj, lk = divmod(idx, 2)
                V[(y_block_row * 2 + lj) * p + (y_block_col * 2 + lk)][cr] += strassen_V[r][idx]
            for idx in range(4):
                li, lk = divmod(idx, 2)
                W[(y_block_col * 2 + lk) * n + li][cr] += strassen_W[r][idx]
    
    for j in range(4):
        add_strassen_block(j * 14, 0, 0, j)
        add_strassen_block(j * 14 + 7, 1, 1, j)
    
    if verify_decomposition(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    return naive_solution()
# EVOLVE-BLOCK-END
