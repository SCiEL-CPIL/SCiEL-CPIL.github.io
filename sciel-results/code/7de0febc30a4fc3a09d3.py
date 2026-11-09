"""Rewritten program: block-wise Kronecker decomposition for 3x5 by 5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
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

    # Block-wise Kronecker decomposition:
    # Treat A (3x5) as 1x3 block row of 1x5 vectors
    # Treat B (5x6) as 5x1 block column of 1x6 vectors
    # The inner product of a 1x5 and 5x1 is a scalar (rank 1 for that block)
    # But we need to handle the full multiplication.
    #
    # Actually, let's think more carefully:
    # C = A * B where A is 3x5, B is 5x6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The standard tensor decomposition has rank n*m*p = 90.
    # 
    # A block approach: group by i (rows of A). For each i, we have a 1x5 row of A
    # and the full 5x6 B. The contribution to C[i,:] is A[i,:] * B (1x6 vector).
    # This is a 1x5 by 5x6 multiplication, which has rank 5 (schoolbook for that sub-problem).
    # So for each of the 3 rows, we need rank 5, total rank = 3*5 = 15.
    #
    # Let's construct this:
    # For row i of A (0-indexed), we decompose A[i,:] * B using 5 terms.
    # Term r (0 to 4) for row i:
    #   U[i*m + j, r] = A[i,j] for j=0..4, but we need U to be fixed...
    #
    # Wait, the tensor decomposition is:
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is the "1" tensor, not the actual multiplication. The actual multiplication
    # tensor has entries T[i,j,k] = 1 for all i,j,k (since A[i,j]*B[j,k] contributes
    # to C[i,k] with coefficient 1).
    #
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    #
    # Block decomposition by i:
    # For each i, we need a rank-5 decomposition of the 5x6 sub-tensor.
    # The sub-tensor for fixed i has entries T[j,k] = 1 for all j,k (5x6 all-ones).
    # The rank of a 5x6 all-ones matrix is 1, but the tensor here is different.
    #
    # Actually, for fixed i, the condition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j,k
    #
    # Let's define for fixed i:
    #   u_i[j, r] = U[i*m+j, r]
    #   v[j, k, r] = V[j*p+k, r]  
    #   w_i[k, r] = W[k*n+i, r]
    #
    # Then: sum_r u_i[j,r] * v[j,k,r] * w_i[k,r] = 1 for all j,k
    #
    # This is a 3-tensor decomposition of the all-ones 5x6 tensor (with the j index
    # appearing in both U and V).
    #
    # A simple rank-5 decomposition for fixed i:
    # For r = 0..4:
    #   u_i[j, r] = 1 if j == r, else 0  (one-hot on j)
    #   v[j, k, r] = 1 if j == r, else 0  (same one-hot)
    #   w_i[k, r] = 1 for all k
    #
    # Then: sum_r u_i[j,r] * v[j,k,r] * w_i[k,r] = sum_r (1 if j==r) * (1 if j==r) * 1
    #   = 1 (only the term r=j contributes, and it's 1*1*1 = 1)
    #
    # This works! And it's rank 5 per i, so total rank = 3*5 = 15.
    #
    # But wait: V is shared across all i. In the above, v[j,k,r] depends on i implicitly
    # through the structure. Let me re-examine.
    #
    # The decomposition must use the SAME V for all i. So V[j*p+k, r] is fixed.
    # For the above to work with shared V, we need:
    #   V[j*p+k, r] = 1 if j == r, else 0, for all k
    #
    # Then for each i:
    #   U[i*m+j, r] = 1 if j == r, else 0
    #   W[k*n+i, r] = 1 for all k
    #
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #   = sum_r (1 if j==r) * (1 if j==r) * 1
    #   = 1 (only r=j contributes)
    #
    # This is valid! Rank = 5 (since r goes 0..4).
    #
    # But wait: U has shape (n*m, rank) = (15, 5). V has shape (m*p, rank) = (30, 5).
    # W has shape (p*n, rank) = (18, 5).
    #
    # Let's construct this explicitly.

    def block_decomposition():
        rank = m  # 5
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # For each i (0 to n-1), j (0 to m-1), k (0 to p-1):
        # U[i*m + j, r] = 1 if j == r else 0
        # V[j*p + k, r] = 1 if j == r else 0
        # W[k*n + i, r] = 1 for all r
        
        for i in range(n):
            for j in range(m):
                for r in range(rank):
                    if j == r:
                        U[i * m + j][r] = 1.0
                        V[j * p + k][r] = 1.0  # BUG: k not defined here
                    # W
                for k in range(p):
                    for r in range(rank):
                        W[k * n + i][r] = 1.0
        
        return {"U": U, "V": V, "W": W}

    # Fix the bug: V should be set independently of i
    def block_decomposition_fixed():
        rank = m  # 5
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # U: for each i, j: U[i*m+j, r] = 1 if j==r else 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][j] = 1.0
        
        # V: for each j, k: V[j*p+k, r] = 1 if j==r else 0
        for j in range(m):
            for k in range(p):
                V[j * p + k][j] = 1.0
        
        # W: for each k, i: W[k*n+i, r] = 1 for all r
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    W[k * n + i][r] = 1.0
        
        return {"U": U, "V": V, "W": W}

    # Validate the block decomposition
    def validate(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try the block decomposition first
    block_result = block_decomposition_fixed()
    if validate(block_result["U"], block_result["V"], block_result["W"]):
        return block_result

    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
