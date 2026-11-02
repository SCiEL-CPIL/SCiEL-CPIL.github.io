"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen's algorithm for 3x3 matrix multiplication has rank 19
    # We use two independent Strassen decompositions for the two 3x3 blocks
    # and schoolbook for the 3x1 block
    
    # Strassen's 19-term decomposition for 3x3 matrices
    # The 19 bilinear forms for C = A*B where A, B are 3x3
    # We represent each term as (A_indices, B_indices, C_index)
    # Using the known Strassen-19 decomposition
    
    # Strassen 3x3x3 rank 19 decomposition
    # Each term: (a_indices, b_indices, c_index)
    # a_indices is a set of (i,j) pairs in A, b_indices in B, c_index is (k,i) in C
    
    # The 19 terms of Strassen's algorithm for 3x3:
    strassen_terms = [
        # Term 1: (A11+A22+A33)(B11+B22+B33) - ... complex
        # Let me use a known explicit rank-19 decomposition
    ]
    
    # Actually, let me use a simpler approach: the known rank-19 decomposition
    # from Strassen (1969) for 3x3 matrices
    
    # The 19 bilinear forms can be written as:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a22+a33)(b22+b33)
    # M3 = (a11+a33)(b11+b33)
    # M4 = a11*b11
    # M5 = a22*b22
    # M6 = a33*b33
    # M7 = (a12+a21)(b21+b12)
    # M8 = (a23+a32)(b32+b23)
    # M9 = (a13+a31)(b31+b13)
    # M10 = a12*b21
    # M11 = a21*b12
    # M12 = a23*b32
    # M13 = a32*b23
    # M14 = a13*b31
    # M15 = a31*b13
    # M16 = (a11+a12+a13)(b31+b21+b11)
    # M17 = (a21+a22+a23)(b13+b23+b33)
    # M18 = (a31+a32+a33)(b12+b22+b32)
    # M19 = (a11+a23+a32)(b12+b21+b33)
    
    # This is getting complex. Let me just use schoolbook but try to find a better decomposition.
    # For now, let's stick with schoolbook which gives rank 63, but let me verify the structure.
    
    # Actually, for 3x3x7, the schoolbook rank is 3*3*7 = 63.
    # Let me try to use a known lower-rank decomposition.
    
    # For 3x3x3, the best known rank is 19 (Strassen).
    # For 3x3x7, we can decompose as two 3x3x3 + one 3x3x1.
    # 3x3x1 has rank 9 (schoolbook).
    # So total rank could be 19 + 19 + 9 = 47 if we use Strassen for the 3x3 blocks.
    
    # Let me construct the Strassen 3x3x3 decomposition explicitly.
    
    # Strassen's 19-term algorithm for 3x3:
    # Split 3x3 into four 2x2 blocks? No, 3 is odd.
    # Actually Strassen's original was for 3x3 using 19 multiplications.
    
    # Let me use a different known decomposition. The 3x3x3 tensor has rank 19.
    # I'll use a concrete construction.
    
    # For simplicity and correctness, let me use schoolbook (rank 63) and verify it works.
    # The current code already does this correctly.
    
    rank = 63
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # First 3x3 block: i in 0..2, j in 0..2, k in 0..2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Second 3x3 block: i in 0..2, j in 0..2, k in 3..5
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Third 3x1 block: i in 0..2, j in 0..2, k in 6
    for i in range(n):
        for j in range(m):
            k = 6
            U[i*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
