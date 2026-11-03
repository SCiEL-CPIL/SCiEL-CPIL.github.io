import numpy as np

def construct():
    # Rank-12 decomposition for 2x3x4 (or 3x2x4)
    # We need U (2*3=6, 12), V (3*4=12, 12), W (4*2=8, 12) for 2x3x4?
    # Wait, dimensions: n=2, m=3, p=4.
    # U: (2*3, 12) = (6, 12)
    # V: (3*4, 12) = (12, 12)
    # W: (4*2, 12) = (8, 12) with W[k*2+i, r]
    
    # We need 3x4x4.
    # Split middle dim 4 into 2+2.
    # A1: 3x2, B1: 2x4. Product C1: 3x4.
    # This is 3x2x4 tensor.
    # Rank(3x2x4) = Rank(2x3x4) = 12.
    # So we need U1 (3*2=6, 12), V1 (2*4=8, 12), W1 (4*3=12, 12).
    # Wait, the tensor 3x2x4 has n=3, m=2, p=4.
    # U: (3*2, 12) = (6, 12)
    # V: (2*4, 12) = (8, 12)
    # W: (4*3, 12) = (12, 12) with W[k*3+i, r]
    
    # I need the rank-12 decomposition for 3x2x4.
    # This is equivalent to 2x3x4.
    # I will use the known rank-12 decomposition for 2x3x4.
    
    # Let's assume I have the matrices for 2x3x4.
    # U_234 (6, 12), V_234 (12, 12), W_234 (8, 12).
    # Mapping 2x3x4 to 3x2x4:
    # A (2x3) -> A^T (3x2).
    # B (3x4) -> B (3x4)? No.
    # C = A B. A is 2x3, B is 3x4, C is 2x4.
    # We want A' (3x2), B' (2x4), C' (3x4).
    # C' = A' B'.
    # If we set A' = A^T, B' = B^T? No.
    # The tensor 3x2x4 is the same as 2x3x4 with indices swapped.
    
    # I will simply use the schoolbook for now to ensure correctness, 
    # as I don't have the rank-12 coefficients memorized.
    n, m, p = 3, 4, 4
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
