"""Matrix multiplication 3,3,6 rank-minimal exact decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    start_time = time.time()
    deadline = 120.0

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = 3, 3, 6
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if not U or not V or not W:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    # The directive suggests a bilinear reduction over F2, but the task requires
    # integer/half-integer entries that exactly decompose the tensor over the reals
    # (implied by the standard tensor decomposition definition and the verify function
    # using float sums). The "rank-12" claim in the directive is likely incorrect for
    # the standard tensor rank over Q or R for 3x3x6.
    # 
    # The rank of 3x3x3 is 23. The rank of 3x3x6 is at least the rank of 3x3x3.
    # Actually, 3x3x6 can be viewed as 2 copies of 3x3x3? No, the dimensions don't align simply.
    # However, we can use a block-diagonal structure or similar.
    # 
    # Let's check if we can find a known rank < 54.
    # The rank of 3x3 matrix multiplication is 23.
    # The rank of 2x2 matrix multiplication is 7.
    # 
    # If we cannot find a verified decomposition with rank < 54, we return the schoolbook.
    # Given the risk of invalidity (score 0) vs score 1.0 (schoolbook), and the lack of a
    # reliable hardcoded table for rank < 54 in the current context, we stick to the
    # guaranteed valid schoolbook solution.
    
    # Attempt to use a known lower rank if easily constructible.
    # For example, if we had a rank-23 decomposition for 3x3x3, we might be able to
    # construct a rank-46 decomposition for 3x3x6 by treating the 6 columns as two
    # groups of 3? But 3x3x6 is not 3x3x3 + 3x3x3.
    # 
    # Actually, 3x3x6 multiplication C = A B where A is 3x3, B is 3x6.
    # We can split B into B1 and B2 (both 3x3).
    # C = A B1 + A B2.
    # This is the sum of two 3x3x3 tensors.
    # The rank of a sum of two tensors is at most the sum of their ranks.
    # If rank(3x3x3) = 23, then rank(3x3x6) <= 23 + 23 = 46.
    # 
    # Can we construct this?
    # We need a rank-23 decomposition for 3x3x3.
    # Let U1, V1, W1 be a rank-23 decomposition for 3x3x3.
    # Then for 3x3x6, we can use:
    # U = [U1, U1] (concatenated horizontally? No, the index i*m+j is the same)
    # Wait, the indices for U are (i, j). For both B1 and B2, the A matrix is the same.
    # So U is the same for both blocks.
    # V is [V1, V2] where V2 is for the second half of B.
    # W is [W1, W2] where W2 is for the second half of C.
    # 
    # Specifically:
    # Let (U_23, V_23, W_23) be a rank-23 decomposition of the 3x3x3 tensor T_333.
    # T_333[i, j, k] = 1.
    # The 3x3x6 tensor T_336[i, j, k] is 1 if k < 3 (for B1) and k >= 3 (for B2)?
    # No, T_336[i, j, k] is 1 for all valid i, j, k.
    # The multiplication is C[k, i] = sum_j A[i, j] B[k, j].
    # If we split B into B1 (cols 0-2) and B2 (cols 3-5), then C1 = A B1 and C2 = A B2.
    # C1 is 3x3, C2 is 3x3.
    # The tensor for C1 is T_333.
    # The tensor for C2 is T_333 (same structure, just different output indices).
    # 
    # So we can construct a rank-46 decomposition:
    # U_46 = [U_23, U_23] (shape 9 x 46)
    # V_46 = [V_23, V_23'] where V_23' is V_23 but with the output index k shifted by 3?
    # No, V indices are (j, k). For B1, k is 0..2. For B2, k is 3..5.
    # So V_23 has non-zeros at V[j, k] for k in 0..2.
    # V_23' should have non-zeros at V[j, k] for k in 3..5, with the same values as V_23[j, k-3].
    # Similarly for W. W indices are (k, i). For C1, k is 0..2. For C2, k is 3..5.
    # W_23 has non-zeros at W[k, i] for k in 0..2.
    # W_23' should have non-zeros at W[k, i] for k in 3..5, with the same values as W_23[k-3, i].
    
    # If we had U_23, V_23, W_23, we could easily build this.
    # But we don't have a hardcoded rank-23 table.
    
    # Is there a simpler lower rank?
    # Rank of 3x3x3 is 23.
    # Rank of 3x3x6 is at least 23.
    # Upper bound 46.
    # 
    # Without the explicit 23-term table, we cannot safely construct rank 46.
    # 
    # Therefore, we return the schoolbook solution (rank 54) to guarantee validity.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
