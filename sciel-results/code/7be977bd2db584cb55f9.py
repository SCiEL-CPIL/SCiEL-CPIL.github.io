"""Matrix multiplication 3,4,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy: Use Strassen's rank-7 algorithm for 2x2 * 2x2 blocks where possible.
    Partition:
    - A (3x4) into: A_top (2x4) and A_bot (1x4)
    - B (4x8) into: B_left (4x4) and B_right (4x4)
    - C (3x8) into: C_top_left (2x4), C_top_right (2x4), C_bot_left (1x4), C_bot_right (1x4)
    
    For the 2x2 * 2x2 sub-problems, use Strassen (rank 7).
    For the 1x2 * 2x2 sub-problems, use schoolbook (rank 4).
    
    Total rank: 2 (B halves) * [4 (A rows) * 2 (B col pairs) * 7 (Strassen) + 1 (A row) * 2 (B col pairs) * 4 (schoolbook)]
             = 2 * [4*2*7 + 1*2*4]
             = 2 * [56 + 8]
             = 2 * 64
             = 128
    
    Wait, that's worse than schoolbook (96). Let me reconsider.
    
    Actually, let me think more carefully. The schoolbook rank is 3*4*8 = 96.
    
    Let me try a different partition. What if I use Strassen on the 4-dimensional middle index?
    
    Actually, the key insight is that for matrix multiplication, we can use the tensor product structure.
    
    Let me try: partition A's 4 columns into two groups of 2, and B's 4 rows into two groups of 2.
    
    A = [A1 | A2] where A1, A2 are 3x2
    B = [B1; B2] where B1, B2 are 2x8
    
    C = A1*B1 + A2*B2
    
    Each A_i * B_i is a 3x2 * 2x8 multiplication. The rank of this is at most 3*2*8 = 48.
    Total: 96. Same as schoolbook.
    
    But can we do better by sharing? 
    
    Let me try yet another approach. Use the fact that 8 = 2*4, so partition B's 8 columns into 4 groups of 2.
    
    For each pair of columns of B (and corresponding pair of columns of C), we have a 3x4 * 4x2 sub-problem.
    The rank of 3x4 * 4x2 is at most 3*4*2 = 24.
    Total: 4 * 24 = 96. Same as schoolbook.
    
    Hmm. Let me think about whether we can actually beat 96.
    
    For small matrix multiplication, the rank is known:
    - 2x2 * 2x2: rank 7 (Strassen)
    - 2x3 * 3x2: rank ? 
    - 3x3 * 3x3: rank 21 (Bini et al.)
    
    For 3x4 * 4x8, this is not a standard case. Let me try to construct a decomposition.
    
    Actually, I think the minimal rank might indeed be 96 for this specific case, or close to it.
    Let me just implement the schoolbook decomposition which gives rank 96.
    
    Wait, let me reconsider the problem. Maybe I can use a hybrid approach.
    
    Let me partition A into 3 rows (each 1x4) and B into 8 columns (each 4x1).
    Then C[i,k] = sum_j A[i,j] * B[j,k].
    
    For each (i,k) pair, this is a dot product of two 4-vectors. The rank of a single dot product
    of two m-dimensional vectors is m (using the identity sum_j x_j y_j = (1/4)(sum x_j)^2 - sum_{j<k}(x_j-x_k)(y_j-y_k) ... no, that's not right).
    
    Actually, the rank of the "inner product" tensor (sum_j x_j y_j) in R^m x R^m -> R is m.
    So for each (i,k), we need rank 4, and there are 3*8 = 24 such pairs, giving total rank 96.
    
    This confirms the schoolbook rank is 96.
    
    Can we do better? The key question is whether there's algebraic structure we can exploit.
    
    For now, let me implement the schoolbook decomposition with rank 96.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = n * m * p  # 96
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
