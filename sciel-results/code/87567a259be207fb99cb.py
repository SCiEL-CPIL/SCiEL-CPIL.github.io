"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to build a low-rank decomposition using Strassen-like or tensor network methods.
    # For 6x6x6, the trivial rank is 216.
    # We can try to find a rank-minimal decomposition by:
    # 1. Starting with the schoolbook rank-216 tensor.
    # 2. Using linear algebra (SVD or similar) on the tensor slices to reduce rank.
    #    However, exact reduction of the matmul tensor rank for 6x6 is an open problem in algebraic complexity,
    #    though we know upper bounds. The best known upper bound for 6x6 is around 170-190 (depending on recent results).
    #    For a deterministic algorithm within 150s, we can try to use a known low-rank scheme.
    #    One simple approach: Use the standard schoolbook but check if we can merge columns via linear dependencies.
    #    The tensor is M_{ijk} = 1 if (i-1)*(m-1) <= k-1 < i*m and (j-1)*(p-1) <= k-1 < j*p ... no, the indices are i,j,k for A[i,j]*B[j,k].
    #    Actually, the tensor T[i,j,k] = 1 if the term A_{i,l} * B_{l,k} is present. 
    #    In the standard indexing, T is a 3-tensor of size n x m x p? No, the rank of the matmul tensor [n x m, m x p, n x p].
    #    Let's represent the tensor as a 3D array T of shape (n*m, m*p, n*p).
    #    T[a, b, c] = 1 if there exists l such that a corresponds to (i,l), b to (l,k), and c to (i,k).
    #    Specifically:
    #    a = i*m + l  => i = a//m, l = a%m
    #    b = l*p + k  => l = b//p, k = b%p
    #    c = k*n + i  => k = c//n, i = c%n
    #    So T[a,b,c] = 1 if (a%m == b//p) and (b%p == c//n) and (a//m == c%n).
    #    Note: The task description says W[k*n+i, r]. This implies the third index of the output is ordered as k*n+i.
    #    So c = k*n + i.
    
    # We can try to compute the rank of this tensor. 
    # Since n,m,p are small (6), the tensor size is 36 x 36 x 36.
    # We can try to find a rank decomposition.
    # A simple heuristic: 
    # 1. Initialize U, V, W with the schoolbook decomposition (rank 216).
    # 2. Try to reduce rank by finding linear dependencies among the columns of U, V, W that can be combined.
    #    This is equivalent to finding a lower rank decomposition of the tensor.
    #    We can use a simple iterative algorithm: 
    #    - Pick a random linear combination of two columns from U, V, W.
    #    - Check if the resulting tensor is still valid.
    #    - This is hard to check efficiently.
    
    # Alternative: Use the known fact that the rank of 6x6x6 matmul is at most 170 (or so).
    # But without a specific algorithm, we might just return the schoolbook.
    # Let's try a simple optimization: 
    # We can use NumPy to check for rank reduction by treating the tensor as a matrix and finding its rank.
    # But the rank of the tensor is not the same as the matrix rank of a flattening.
    
    # Given the time limit and the difficulty, a safe bet is to return the schoolbook decomposition,
    # but let's see if we can do slightly better with a simple heuristic.
    # For 6x6, we can use a block decomposition with 2x2 blocks.
    # There are 3x3 = 9 blocks of 2x2 matrices.
    # Strassen on 2x2 matrices has rank 7.
    # So total rank would be 9 * 7 = 63? No, that's if the blocks are independent.
    # But the blocks interact.
    # Actually, if we use Strassen for each 2x2 block, we still need to combine them.
    # This is getting complicated.
    
    # Let's stick to a robust method: 
    # We will construct the schoolbook decomposition and then try to reduce the rank by 
    # identifying and removing redundant columns.
    # However, for the matmul tensor, all 216 columns are linearly independent in the 
    # standard representation? No, the rank is much lower.
    # The rank of the 6x6x6 matmul tensor is known to be at most 170 (Coppersmith-Winograd improvements).
    # But implementing Coppersmith-Winograd is complex.
    
    # Let's try a simple SVD-based rank reduction on the tensor slices.
    # We reshape the tensor into a matrix and find its rank.
    # T is 36x36x36. We can reshape to 36 x (36*36).
    # The rank of this matrix gives an upper bound on the tensor rank? No, the tensor rank is 
    # the minimum r such that T = sum r_i u_i v_i w_i.
    # This is the tensor rank, which is NP-hard to compute.
    
    # Given the constraints, I will return the schoolbook decomposition, but I will 
    # add a check to see if we can reduce it by a small amount using a simple heuristic.
    # For now, let's just return the schoolbook to ensure correctness and avoid timeouts.
    
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
