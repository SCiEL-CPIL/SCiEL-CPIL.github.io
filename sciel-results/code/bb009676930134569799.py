"""Program for matrix multiplication 2x3x9 with reduced rank."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9

    # We use a Strassen-like decomposition for the 2x2x2 block structure
    # combined with schoolbook for the remaining parts.
    # 
    # The idea: split the 3x9 matrix B into three 1x9 blocks (rows), and 
    # the 2x3 matrix A into a 2x1 block (col 0) and 2x2 block (cols 1,2).
    # 
    # Actually, let's use a more systematic approach:
    # For 2xm x mxp, we can use the following construction:
    # - Use Strassen on a 2x2x2 sub-problem embedded in the computation
    # - This saves 1 multiplication compared to schoolbook for that sub-block
    #
    # Let me implement a concrete decomposition with rank < 54.
    #
    # Strategy: 
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We'll use the following trick for pairs of columns in B:
    # For each pair of output columns (k, k+1), and for the 2x2 subproblem 
    # involving A[:, 1:3] (a 2x2 matrix) and B[1:3, k:k+2] (a 2x2 matrix),
    # we can use Strassen's 7 multiplications instead of 4.
    #
    # But we need to be careful about how the terms combine.
    #
    # Let me try a different, more reliable approach:
    # Use the known result that for 2xm x mxp, rank <= 2m*p - 2*p + 2*m - 2
    # No, that's not right either.
    #
    # Let me just implement a Strassen-based decomposition for the full problem.
    #
    # Actually, the simplest improvement: use rank 53 by finding one linear 
    # dependency among the 54 schoolbook terms.
    #
    # For 2x3x9, the tensor has a specific structure. Let me think about 
    # whether there's a rank-53 decomposition.
    #
    # The matrix multiplication tensor M_{2,3,9} has rank at most 54 (schoolbook).
    # The border rank is at most 54. The actual rank might be less.
    #
    # Known results: rank of 2x2x2 is 7 (Strassen).
    # For 2x2xp, the rank is 7p - 2p + 2 = 5p + 2? No.
    #
    # Let me just implement Strassen for a 2x2x2 block and schoolbook for the rest.
    #
    # Split A into [A0 | A1] where A0 is 2x1 (first column) and A1 is 2x2 (last two columns).
    # Split B into [B0; B1] where B0 is 1x9 (first row) and B1 is 2x9 (last two rows).
    #
    # C = A0 * B0 + A1 * B1
    #
    # A0 * B0: 2x1 times 1x9 = 2x9. This requires 2*1*9 = 18 multiplications.
    # A1 * B1: 2x2 times 2x9. We can use Strassen here.
    #
    # For 2x2 times 2x9:
    # Split B1 into two 2x4 and one 2x1? Or use Strassen column by column?
    #
    # Actually, for 2x2 times 2xp, we can use 7p - 2p + 2 = 5p + 2? 
    # No, let me think more carefully.
    #
    # For 2x2 times 2xp, we can process each column independently with Strassen,
    # but that gives 7p multiplications. We can do better by sharing.
    #
    # Actually, the rank of 2x2xp is 7p - 2p + 2 = 5p + 2 for p >= 2? 
    # Let me check: for p=1, rank is 7. 5*1+2 = 7. Yes.
    # For p=2, rank is 12? 5*2+2 = 12. Let me verify... 
    # Actually, I'm not sure about this formula.
    #
    # Let me just use schoolbook for A1*B1: 2*2*9 = 36 multiplications.
    # Total: 18 + 36 = 54. That's the same as full schoolbook.
    #
    # To save multiplications, I need to actually use Strassen for the 2x2x9 part.
    #
    # For 2x2 times 2xp using Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B11)
    # M4 = A22(B12 - B22)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # For each column k of the output:
    # C11,k = M1 - M4 - M5 - M7
    # C12,k = M3 + M5
    # C21,k = M2 + M4
    # C22,k = M1 - M2 - M3 - M6
    #
    # But M1..M7 depend on all columns of B1. So we need 7*p multiplications total.
    # That's 7*9 = 63, which is worse than 36.
    #
    # Wait, no. Each M_i is a 1x9 vector (or we compute it per column).
    # Actually, M1 = (A11+A22) * (B11_row + B22_row) where B11_row and B22_row are 1x9.
    # So M1 is a 1x9 vector, computed with 9 multiplications (one per column).
    # Total: 7*9 = 63. That's worse.
    #
    # The issue is that Strassen doesn't help when one dimension is 2 and another is large,
    # unless we can share computations across columns.
    #
    # Let me abandon this approach and just use schoolbook rank 54, but ensure correctness.
    
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
# EVOLVE-BLOCK-END
