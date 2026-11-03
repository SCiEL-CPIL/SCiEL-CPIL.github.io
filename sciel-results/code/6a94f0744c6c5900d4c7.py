"""Program for matrix multiplication 3x4x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # We aim for rank 36 using a cyclic polynomial embedding approach.
    # This splits the problem into smaller polynomial multiplications.
    
    rank = 36
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # This is a placeholder for the actual rank-36 decomposition.
    # A valid rank-36 decomposition for 3x4x6 is non-trivial.
    # For the sake of providing a valid program, we fall back to schoolbook 
    # if the rank-36 construction is not verified.
    # However, to attempt the improvement, we use a structured decomposition.
    
    # Let's use the splitting strategy:
    # C = A B
    # Split B into B1 (4x3) and B2 (4x3)
    # C1 = A B1, C2 = A B2
    # For each, split A into A1 (3x2) and A2 (3x2)
    # C1 = A1 B1_1 + A2 B1_2 (where B1_1 is first 2 cols of B1, etc? No, B1 is 4x3, A is 3x4.
    # A (3x4) * B1 (4x3).
    # Split A into A_left (3x2) and A_right (3x2).
    # Split B1 into B1_top (2x3) and B1_bottom (2x3).
    # C1 = A_left B1_top + A_right B1_bottom.
    # Each term is 3x2 * 2x3.
    # We need a rank-9 decomposition for 3x2 * 2x3.
    
    # If we cannot find rank 9, we use rank 18 (schoolbook) for each.
    # Total rank 4 * 18 = 72.
    
    # Since I cannot derive the rank-9 decomposition, I will return the schoolbook 
    # decomposition to ensure correctness.
    
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
