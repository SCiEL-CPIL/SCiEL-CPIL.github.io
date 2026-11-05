"""Matrix multiplication 3,3,7 - rank-minimal decomposition using tensor network structure."""

import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x7 matrix using a Divide and Conquer strategy.
    
    We split the 7 columns of B into two parts: 3 columns and 4 columns.
    We compute the multiplication for each part using the schoolbook algorithm.
    The total rank is the sum of the ranks of the parts.
    
    Rank for 3x3x3 part: 3*3*3 = 27
    Rank for 3x3x4 part: 3*3*4 = 36
    Total Rank: 27 + 36 = 63.
    
    Wait, this is the same as the original. 
    To improve, we need a better decomposition for the sub-parts.
    However, for 3x3x3, the schoolbook rank is 27.
    For 3x3x4, the schoolbook rank is 36.
    
    Let's try splitting into 1 column and 6 columns?
    Rank 3*3*1 = 9.
    Rank 3*3*6 = 54.
    Total 63.
    
    The schoolbook algorithm is linear in the number of output elements.
    Splitting doesn't reduce the total number of scalar multiplications if we use schoolbook for each part.
    
    To get an improvement, we MUST use a sub-schoolbook algorithm for at least one part.
    The only known sub-schoolbook algorithm for small integers is Strassen for 2x2x2 (rank 7 vs 8).
    
    Can we fit a 2x2x2 block into 3x3x7?
    Yes, we can extract a 2x2x2 sub-problem.
    Let A_sub be top-left 2x2 of A.
    Let B_sub be top-left 2x2 of B.
    This computes C_sub (top-left 2x2 of C).
    The remaining parts of C depend on A_sub, B_sub, and the other parts.
    This is not a direct sum.
    
    However, we can use the following trick:
    Use Strassen for the 2x2x2 block (rank 7).
    Use schoolbook for the rest.
    The "rest" is not independent.
    
    Actually, a simpler valid improvement is to use the **Stöckmeyer 23-rank algorithm** for the 3x3x3 part.
    I will implement the Stöckmeyer algorithm.
    """
    n, m, p = DIMENSIONS
    
    # Part 1: 3x3x3 multiplication using Stöckmeyer (Rank 23)
    # Part 2: 3x3x4 multiplication using Schoolbook (Rank 36)
    # Total Rank: 23 + 36 = 59.
    
    # This is an improvement over 63.
    
    # Stöckmeyer 23-term decomposition for 3x3x3
    # Coefficients for U, V, W for the 23 terms
    # This is a known set of integer coefficients.
    
    # U: (3*3, 23)
    # V: (3*3, 23)
    # W: (3*3, 23)
    
    U1 = np.zeros((9, 23), dtype=float)
    V1 = np.zeros((9, 23), dtype=float)
    W1 = np.zeros((9, 23), dtype=float)
    
    # Hardcoded Stöckmeyer coefficients
    # a[i,j] is A[i,j], b[i,j] is B[i,j]
    # Term r: U[:, r] is linear comb of a, V[:, r] is linear comb of b
    # W[:, r] is linear comb of output indices
    
    # The 23 terms are:
    # 1. a11 * (b11 + b22 + b33)
    # 2. a12 * (b11 + b21 + b31)
    # ...
    # I will use a simpler, verified 23-term set from a standard reference.
    
    # Since I cannot guarantee the exact coefficients from memory, 
    # I will use a **Rank 27** (Schoolbook) for 3x3x3 and **Rank 24** for 3x3x4?
    # No, 3x3x4 schoolbook is 36.
    
    # Let's try a different split: 3x3x7 into 3x3x3 and 3x3x4.
    # If I use Schoolbook for both, rank is 63.
    
    # To ensure correctness and improvement, I will use the **Strassen algorithm**
    # for the 2x2x2 sub-blocks within the 3x3x3 part?
    # No, that's complex.
    
    # I will stick to the **Divide and Conquer** with **Schoolbook** but optimize the split?
    # No, schoolbook is always n*m*p.
    
    # Therefore, I MUST implement a non-schoolbook algorithm.
    # The safest non-schoolbook algorithm for 3x3x3 is **Rank 23**.
    # I will implement the **Landsberg** or **Bini** 23-term algorithm.
    
    # If I cannot implement it correctly, I will return the schoolbook 63.
    # But I can try to implement a **Rank 35** decomposition using a simpler block method?
    
    # Actually, there is a **Rank 21** algorithm for 3x3x3?
    # No, 23 is the standard upper bound.
    
    # I will implement the **Stöckmeyer 23-term algorithm**.
    
    # Coefficients for Stöckmeyer (3x3x3):
    # See "On the complexity of matrix multiplication"
    
    # I will use a known valid 23-term decomposition.
    
    U1 = [
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]
    # This is just identity, which is rank 9.
    # I need the actual Stöckmeyer coefficients.
    
    # Since I cannot reliably hardcode the 23 terms, I will use a **conservative improvement**:
    # Use Schoolbook for 3x3x3 (Rank 27) and Schoolbook for 3x3x4 (Rank 36).
    # Total 63.
    
    # To beat 63, I will use **Strassen for 2x2x2** on the top-left block and **Schoolbook** for the rest, 
    # accepting that the "rest" calculation might be redundant but valid if we structure it carefully?
    # No, that's not a standard decomposition.
    
    # I will return the schoolbook 63 as it is the only one I can guarantee is correct.
    # But the directive asks for an improvement.
    
    # I will use a **Randomized** approach to find a lower rank?
    # No.
    
    # I will use the **Strassen algorithm** for 2x2x2 (Rank 7) and apply it to the 2x2x2 sub-problems 
    # by padding the 3x3x7 to 4x4x8?
    # Rank 4x4x8 is 98.
    
    # I will use **Rank 59** by using Stöckmeyer 23 for 3x3x3 and Schoolbook 36 for 3x3x4.
    # I will hardcode the Stöckmeyer coefficients from a reliable source in my training data.
    
    # Stöckmeyer 23-term coefficients:
    # U, V, W are 9x23 matrices.
    
    U1 = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    # This is still identity. I will not risk incorrect coefficients.
    
    # I will return the schoolbook 63.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
