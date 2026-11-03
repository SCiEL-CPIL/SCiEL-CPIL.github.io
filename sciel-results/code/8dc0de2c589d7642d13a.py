"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for multiplying a 3x3 matrix A by a 3x8 matrix B.
    
    Strategy: Use Strassen's algorithm (rank 7) for 2x2x2 sub-blocks.
    Partition A (3x3) and B (3x8) into blocks:
      A = [[A11, A12, A13],   B = [[B11, B12, B13, B14],
           [A21, A22, A23],        [B21, B22, B23, B24],
           [A31, A32, A33]]        [B31, B32, B33, B34]]
    
    We compute C = A * B where C is 3x8.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For each output column k (0..7), C[:,k] = A * B[:,k]
    
    Approach: For each of the 8 columns of B, compute A * B_col using a rank-19 or lower decomposition.
    But we can do better by sharing structure across columns.
    
    Let's use a direct approach: decompose the full 3x3x8 multiplication tensor.
    
    We'll use Strassen on 2x2 blocks. Partition:
    - Top 2 rows of A, all 3 columns: A_top (2x3)
    - Bottom 1 row of A, all 3 columns: A_bot (1x3)
    - Top 2 rows of B, all 8 columns: B_top (2x8)  
    - Bottom 1 row of B, all 8 columns: B_bot (1x8)
    
    C_top = A_top * B_top + A_top * B_bot  (2x8)
    C_bot = A_bot * B_top + A_bot * B_bot  (1x8)
    
    For C_top: A_top is 2x3, B_top is 2x8, B_bot is 1x8
    C_top = A_top * [B_top; B_bot]  -- no, that's not right.
    
    Actually: C_top[i,k] = A_top[i,0]*B_top[0,k] + A_top[i,1]*B_top[1,k] + A_top[i,2]*B_bot[0,k]
    
    Let me use a different partition. For each pair of consecutive columns in B:
    
    Actually, the simplest correct approach: use schoolbook for the full 3x3x8.
    Rank = 3*3*8 = 72.
    
    To improve, note that for each column k of B, we compute a 3-vector C[:,k] = A * B[:,k].
    The tensor is a direct sum of 8 copies of the 3x3x3 multiplication tensor (one per column).
    
    So rank(3x3x8) = 8 * rank(3x3x3).
    
    The best known rank for 3x3x3 is 19 (achieved by various algorithms).
    So rank(3x3x8) <= 8 * 19 = 152. That's worse than 72!
    
    Wait, that's not right. The direct sum rank is at most the sum of ranks, but we can share.
    Actually no - for a direct sum of tensors, the rank is at most the sum, but can be less.
    However, for matrix multiplication tensors, the direct sum of n copies has rank at most n * rank(single).
    
    But 72 < 152, so schoolbook is better than using rank-19 for each column independently.
    
    Let me think differently. The 3x3x8 tensor T satisfies:
    T[i,j,k] = delta(i,j,k) in the sense that T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the multiplication is C[i,k] = sum_j A[i,j]B[j,k].
    
    The standard decomposition has rank n*m*p = 72.
    
    Can we do better? For 3x3x3, we know rank <= 23 (Laderman) and >= 19.
    For 3x3x8, we can write it as a block matrix multiplication.
    
    Let me try: partition B into four 3x2 blocks: B = [B1, B2, B3, B4] where each Bj is 3x2.
    C = [A*B1, A*B2, A*B3, A*B4]
    
    For each 3x3x2 subproblem, what's the minimum rank?
    Schoolbook: 3*3*2 = 18.
    
    Can we do better for 3x3x2? The tensor is C[i,k] = sum_j A[i,j]*B[j,k] for i=0..2, j=0..2, k=0..1.
    
    This is like multiplying a 3x3 by a 3x2. We can view this as:
    For each k (0,1), C[:,k] = A * B[:,k]
    
    Using Strassen-like ideas on 2x2 sub-blocks of A:
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    
    For the top 2x2 block of A and all of B:
    C_top = A_top * B where A_top is 2x3, B is 3x2.
    
    Hmm, let me just implement a verified Strassen-based decomposition.
    
    Actually, let me use a simpler improvement: for the 3x3x2 subproblem, I can use
    a rank-14 or lower decomposition. But I'm not sure of the exact minimum.
    
    Let me just use schoolbook (rank 72) which is correct and safe.
    """
    n, m, p = 3, 3, 8
    rank = n * m * p  # 72
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
# EVOLVE-BLOCK-END
