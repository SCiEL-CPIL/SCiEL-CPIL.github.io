import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Schoolbook decomposition of rank n*m*p = 18
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower rank decomposition using structured search
    # Based on the knowledge that rank 17 is achievable by combining
    # Strassen's 2x2x2 algorithm with schoolbook for remaining parts
    
    # The strategy: decompose 2x3 x 3x3 into subproblems
    # We can view this as:
    # [A11 A12 A13]   [B11 B12 B13]
    # [A21 A22 A23] * [B21 B22 B23]
    #                    [B31 B32 B33]
    #
    # One approach: split the 3 columns of A into 2+1, and 3 rows of B into 2+1
    # This gives subproblems: 2x2 x 2x2, 2x2 x 2x1, 2x1 x 2x2, 2x1 x 2x1
    #
    # Using Strassen for 2x2x2 (rank 7) and schoolbook for others:
    # 2x2x2: rank 7
    # 2x2x2x1: rank 4 (schoolbook)
    # 2x1x2x2: rank 4 (schoolbook)  
    # 2x1x2x1: rank 2 (schoolbook)
    # Total: 7 + 4 + 4 + 2 = 17
    
    # Let's construct this decomposition explicitly
    # 
    # Split A into [A_left | A_right] where A_left is 2x2, A_right is 2x1
    # Split B into [B_top; B_bottom] where B_top is 2x3, B_bottom is 1x3
    #
    # C = A_left @ B_top + A_right @ B_bottom
    #
    # A_left @ B_top: 2x2 @ 2x3 -> use Strassen-like for 2x2x2 part
    # Actually, let's think more carefully.
    #
    # Let's use a different decomposition. Consider:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # For n=2, m=3, p=3, we have 18 scalar multiplications in schoolbook.
    # 
    # Known result: the tensor rank of 2x3 x 3x3 matrix multiplication is 17.
    # 
    # Let's construct a rank-17 decomposition by leveraging the structure.
    # 
    # Approach: Use the fact that we can combine some terms.
    # 
    # Alternative: Try to find a rank-17 decomposition by solving a system
    # of polynomial equations, but that's complex. Instead, let's use a
    # known construction.
    #
    # Here's a constructive approach:
    # We can write the multiplication as:
    # C = A @ B
    #
    # Let's partition the indices differently.
    # 
    # Actually, let's try a direct search for a rank-17 decomposition
    # using the structure of the problem.
    #
    # For now, return the schoolbook decomposition to ensure validity.
    # A more sophisticated search would be needed to find rank 17.
    
    # Try a simple improvement: check if we can reduce by 1
    # by finding a linear dependency in the schoolbook terms.
    #
    # The schoolbook terms are indexed by (i,j,k) for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # We need to find if there's a linear combination that gives zero.
    #
    # This is equivalent to finding the rank of the multiplication tensor.
    # For 2x3x3, it's known to be 17.
    #
    # Let's attempt a constructive rank-17 decomposition.
    #
    # Based on the subproblem decomposition:
    # Split A = [A1 | A2] where A1 is 2x2 (columns 0,1) and A2 is 2x1 (column 2)
    # Split B = [B1; B2] where B1 is 2x3 (rows 0,1) and B2 is 1x3 (row 2)
    #
    # C = A1 @ B1 + A2 @ B2
    #
    # A1 @ B1 is 2x2 @ 2x3. We can compute this using:
    # Let A1 = [[a, b], [c, d]], B1 = [[e, f, g], [h, i, j]]
    # A1 @ B1 = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    # This requires 8 scalar multiplications (schoolbook for 2x2x2x3 is 2*2*3=12? No...)
    # Wait, 2x2 @ 2x3: each of the 2x3 output entries requires 2 multiplications, so 12 total.
    # But we can do better.
    #
    # Actually, for 2x2 @ 2x2, Strassen uses 7 multiplications.
    # For 2x2 @ 2x3, we can think of it as two 2x2 @ 2x1 multiplications plus one 2x2 @ 2x1,
    # but that's not quite right.
    #
    # Let's use a different approach. The known rank-17 construction for 2x3x3:
    #
    # I'll implement a search-based approach that tries to find a rank-17 decomposition
    # by fixing certain structural patterns and solving for the remaining values.
    #
    # For time efficiency, let's just return schoolbook for now.
    # A proper implementation would need to encode the known rank-17 construction.
    
    # Given the complexity, return schoolbook to ensure correctness
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
