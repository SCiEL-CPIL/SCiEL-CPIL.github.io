"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

import numpy as np
import time

def construct():
    """
    Constructs a decomposition for 2x4 * 4x7 matrix multiplication.
    
    Strategy:
    1. Use the Strassen decomposition for the 4x4 * 4x4 multiplication (rank 7 instead of 16).
    2. Combine with the remaining 4x3 * 4x3 multiplication (rank 12 schoolbook or potentially less, but let's start with schoolbook for the 3x3 part? No, wait).
    
    Let's analyze the structure:
    A is 2x4, B is 4x7.
    A = [A1 | A2] where A1 is 2x4? No, A is 2x4.
    Let's partition B into two 4x4 and one 4x3 block?
    B = [B1 | B2] where B1 is 4x4, B2 is 4x3.
    Then AB = A B1 + A B2.
    
    Wait, the multiplication is A (2x4) * B (4x7).
    We can view A as 2x4.
    We can view B as 4x7.
    
    If we just do schoolbook, rank = 2*4*7 = 56.
    
    The hint mentions a "rank 50" solution.
    56 - 50 = 6. So we save 6 multiplications.
    
    How to save multiplications?
    The middle dimension is 4.
    Standard Strassen works for 2x2 * 2x2 saving 1 multiplication (8 vs 8? No, 7 vs 8).
    For 4x4 * 4x4, Strassen gives rank 7 (using 3x3? No, 4x4 is 2^2, so 7^2 = 49? No, rank of 4x4*4x4 is 49? No, schoolbook is 64. Strassen is 7^2=49? Yes.
    But our A is 2x4, not 4x4.
    
    Let's try to embed the 4x4 Strassen.
    Pad A to 4x4 with zeros.
    A_pad = [A; 0; 0; 0] (4x4).
    B is 4x7. Pad B to 4x4? No, we need the full 7 columns.
    
    Let's split B into B_left (4x4) and B_right (4x3).
    AB = A B_left + A B_right.
    
    For A B_left (2x4 * 4x4):
    Pad A to 4x4 as A_p = [A; Z] where Z is 2x4 zeros.
    Then C_p = A_p * B_left is 4x4.
    The top-left 2x4 block of C_p is exactly A B_left.
    If we use Strassen for A_p * B_left, the rank is 7 (for 4x4? No, Strassen for 2x2 is 7. For 4x4, it's 7^2 = 49? No, the recursion is: M_4 = 7 M_2 = 7 * 8 = 56? No.
    Rank of 4x4*4x4 tensor.
    Schoolbook: 64.
    Strassen: 7 (for 2x2) -> 7^2 = 49 for 4x4?
    Wait, the rank of the 4x4*4x4 multiplication tensor is 49?
    Actually, the rank of 4x4*4x4 is 49?
    Let's check: 2x2 rank is 7. 4x4 rank is 7^2 = 49.
    8x8 rank is 7^3 = 343.
    So using Strassen for the 4x4 part gives rank 49 for that sub-problem.
    
    For A B_right (2x4 * 4x3):
    Schoolbook rank is 2*4*3 = 24.
    Total rank = 49 + 24 = 73.
    This is worse than 56.
    
    So Strassen on the 4x4 part is not helpful for the 2x4 case directly because the "pad" rows add cost.
    
    However, the hint says rank 50 is possible.
    56 - 50 = 6.
    Maybe there is a specific decomposition for 2x4*4x7?
    
    Let's look at the structure of 2x4 * 4x7.
    It can be seen as two 1x4 * 4x7 multiplications.
    Rank of 1x4 * 4x7 is 4*7 = 28? No, 1*4*7 = 28.
    Two such terms would be 56.
    But they share the same B?
    A = [a1; a2].
    C = a1 B + a2 B.
    This is linear in A.
    Is it possible to share computation between the two rows of A?
    Yes, if we group them.
    
    Let's try to find a decomposition of rank 50.
    50 = 49 + 1?
    Or 50 = 56 - 6?
    
    Maybe the 4x4 part can be done with rank 48?
    Or the 4x3 part with rank 18?
    
    Let's try a different split.
    Split A into A1 (2x2) and A2 (2x2).
    Split B into B1 (4x4) and B2 (4x3).
    This doesn't help directly.
    
    What if we use the fact that 2x4 * 4x7 is a sub-problem of 4x4 * 4x7?
    Rank of 4x4 * 4x7:
    4x4 * 4x4 is rank 49.
    4x4 * 4x3 is rank 4*4*3 = 48.
    Total 97.
    
    Let's go back to the hint: "Constructs a verified rank-50 Strassen-based decomposition".
    This implies Strassen is involved.
    
    Perhaps the decomposition is not a simple sum of sub-problems.
    Maybe it's a tensor decomposition of the full 2x4*4x7 tensor.
    
    Let's try to construct a rank 50 decomposition.
    I will implement a search for a rank 50 decomposition using linear algebra if possible, or fallback to 56.
    
    Actually, I can try to hardcode a known decomposition if I can derive it.
    But deriving it is hard.
    
    Alternative:
    The score is seed/value.
    Seed is 56.
    If I return rank 50, score = 56/50 = 1.12.
    If I return rank 56, score = 1.0.
    
    I will attempt to find a rank 50 decomposition by solving a system of linear equations?
    The number of variables is 2*4*50 + 4*7*50 + 7*2*50 = 400 + 1400 + 700 = 2500.
    The number of constraints is 2*4*7 = 56? No, the tensor has 56 entries, but the decomposition must satisfy the identity for all inputs.
    The identity is: sum_c U[i,j,c] V[j,k,c] W[k,l,c] = delta(i,j) * delta(k,l)? No.
    The tensor is T[i,j,k] = delta_{j,k} ? No.
    The multiplication tensor is T_{i,j,k} = delta_{j,k} if we index A by i,j and B by k,l?
    Wait, A is 2x4, B is 4x7.
    C[i,l] = sum_j A[i,j] B[j,l].
    The tensor T has indices i, j, l.
    T[i,j,l] = 1 if j is the contraction index?
    No, the tensor is T[i,j,l] such that C[i,l] = sum_j T[i,j,l] A[i,j] B[j,l].
    This is not a standard tensor decomposition form.
    The standard form is C[i,l] = sum_c U[i,c] V[c,l].
    Here, the "inputs" are A and B.
    The tensor is M[i,j,l] where M[i,j,l] = delta_{j,l}? No.
    The multiplication map is multilinear.
    C_{i,l} = sum_j A_{i,j} B_{j,l}.
    This can be viewed as a tensor T in V_A^* \otimes V_B^* \otimes V_C.
    T = sum_{i,j,l} e_i \otimes e_j \otimes f_l * (e_i^* \otimes e_j^* \otimes f_l^*)?
    Actually, the tensor is T = sum_{i,j,l} E_{i,j} \otimes E_{j,l} \otimes E_{i,l}^*?
    No.
    The tensor is T = sum_{i,j,l} A_{i,j} B_{j,l} -> C_{i,l}.
    In terms of basis, T = sum_{i,j,l} E_{i,j}^{(A)} \otimes E_{j,l}^{(B)} \otimes E_{i,l}^{(C)*}.
    We want to write T = sum_c u_c \otimes v_c \otimes w_c.
    Where u_c is in (R^2 x R^4)^*, v_c in (R^4 x R^7)^*, w_c in (R^2 x R^7)^*.
    This means u_c = U_c (2x4), v_c = V_c (4x7), w_c = W_c (2x7).
    The condition is: for all A, B:
    sum_c (A U_c) (V_c B) = A B.
    Here (A U_c) is 2x1? No.
    U_c is 2x4? No, U is (n*m, rank) = (8, R).
    So U_c is a vector of length 8.
    A is 2x4. Flatten A to a vector a of length 8.
    Then (A U_c) is scalar a^T u_c.
    Similarly v_c is length 28. b^T v_c.
    w_c is length 14.
    So C = sum_c (a^T u_c) (b^T v_c) w_c^T?
    No, C is 2x7.
    The output is C.
    The decomposition is C = sum_c (a^T u_c) (b^T v_c) w_c.
    Where w_c is a 2x7 matrix? No, w_c is a vector of length 14 in the code's W matrix.
    W is (p*n, rank) = (14, R).
    So w_c is the c-th column of W, which is a 14-vector.
    Reshaping w_c to 2x7 gives the matrix W_c.
    
    So the condition is:
    For all a, b:
    sum_c (a^T u_c) (b^T v_c) w_c = a B? No.
    The equation is:
    For all A, B:
    sum_c (flatten(A) \cdot u_c) (flatten(B) \cdot v_c) reshape(w_c) = A B.
    
    This is a system of polynomial equations.
    We can solve it by linear algebra if we fix the structure?
    Or we can try to find a rank 50 solution by random search or optimization.
    
    Given the time limit (120s), I can't do heavy optimization.
    I will check if rank 50 is achievable with a known pattern.
    
    Pattern:
    Use Strassen for the 4x4 part (rank 49) and add 1 for the 4x3 part?
    Total 50.
    How?
    If the 4x3 part can be absorbed?
    
    Let's try to construct the 49+1 decomposition.
    A (2x4). B (4x7).
    Let B = [B1 (4x4), B2 (4x3)].
    C = A B1 + A B2.
    
    If we use Strassen for A B1, we need to embed A in 4x4.
    Let A_p = [A; 0; 0; 0].
    Let B1 be 4x4.
    C_p = A_p B1 (4x4).
    The top 2 rows of C_p are A B1.
    The rank of A_p B1 is 49.
    The matrices are:
    U_p: 16x49.
    V_p: 16x49.
    W_p: 16x49.
    We only need the top 2 rows of U_p? No, U_p corresponds to A_p.
    A_p is 4x4. A is the top 2 rows.
    So U_p has 16 rows. The first 8 rows correspond to A.
    The last 8 rows are for the zeros.
    If we set the last 8 rows of U_p to 0, does it work?
    Then the contribution is sum_c (a_p^T u_c) ...
    If a_p = [a; 0], then a_p^T u_c = a^T u_c[0:8].
    So we can just use the first 8 rows of U_p.
    Similarly for V_p? V_p corresponds to B1. B1 is 4x4.
    B1 is part of B.
    B is 4x7. B1 is the first 4 columns.
    So V_p should be 16x49? No, V is (m*p, rank) = (28, R).
    V_p for B1 would be 16x49 (4x4 matrix).
    But B has 7 columns.
    We need to handle the other 3 columns.
    
    Let's assume the decomposition is:
    Part 1: Rank 49 for A B1.
    Part 2: Rank 1 for A B2?
    If Part 2 has rank 1, then A B2 = (a^T u) (b^T v) w.
    This implies A B2 is rank 1 for all A, B?
    That's impossible.
    
    So the rank 50 must be distributed differently.
    
    Maybe the 4x4 part has rank 48?
    Or the 4x3 part has rank 2?
    
    I will fallback to 56 if I can't find 50 quickly.
    But I will try a simple search for rank 50.
    """
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    
    # Schoolbook
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to find a better rank?
    # Given time constraints, I will return schoolbook.
    # However, the prompt strongly hints at 50.
    # I will return schoolbook to be safe, as I cannot guarantee 50.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
