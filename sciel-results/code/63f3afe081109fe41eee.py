"""Matrix multiplication 3,4,8 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 8
    start_time = time.time()
    deadline = 140  # 10s safety margin from 150s limit
    
    # Strassen-style decomposition:
    # 1) 3x3 Strassen on the row blocks (rank 23)
    # 2) 2x2 Strassen on the column blocks (rank 7)
    # Total rank = 23 * 7 = 161
    
    # 3x3 Strassen coefficients for combining 3 sub-blocks:
    # M1 = (A1 + A2)(B1 + B2)  -> coeff 1
    # M2 = A1 * (B2 - B1)      -> coeff for B1: -1, B2: 1
    # M3 = (A1 + A2 + A3) * B3 -> coeff 1
    # M4 = A3 * (B3 - B2)      -> coeff for B2: -1, B3: 1
    # M5 = A2 * (B2 - B3)      -> coeff for B2: 1, B3: -1
    # M6 = (A2 + A3) * (B3 - B1) -> coeff for B1: -1, B3: 1
    # M7 = (A1 - A2) * (B1 + B2) -> coeff for B1: 1, B2: 1
    
    # For 3x3 Strassen, the output blocks C1, C2, C3 are:
    # C1 = M1 - M2 + M3 - M4
    # C2 = M1 + M2 + M4 - M5
    # C3 = M3 - M1 + M5 + M6 - M7
    
    # 2x2 Strassen for 4x4 blocks (splitting 4 columns into 2+2):
    # M1 = A * (B1 + B2)
    # M2 = (A1 - A2) * B2
    # M3 = (A1 + A2) * B1
    # M4 = A * (B2 - B1)
    # C1 = M3 + M4
    # C2 = M1 - M4
    
    # We'll build the decomposition by combining these.
    # The tensor has indices (i, j, k) where:
    # i in {0,1,2} (rows of A, rows of C)
    # j in {0,1,2,3} (cols of A, rows of B)  
    # k in {0,...,7} (cols of B, cols of C)
    
    # W convention: W[k*n + i, r]
    # U convention: U[i*m + j, r]
    # V convention: V[j*p + k, r]
    
    # For 3x3 Strassen, we have 7 intermediate products M1-M7
    # For 2x2 Strassen, each M is further decomposed into 4 sub-products
    
    # Total rank = 7 * 7 = 49? No wait.
    # 3x3 Strassen: 7 multiplications
    # 2x2 Strassen: 4 multiplications per 4x4 block
    # So total rank = 7 * 4 = 28? 
    
    # Actually, let me reconsider. The 3x3 Strassen gives 7 "scalar" multiplications
    # of 4x4 matrices. Each 4x4 multiplication via 2x2 Strassen gives 4 "scalar" multiplications
    # of 2x2 matrices. But 2x2 Strassen has rank 7, not 4.
    
    # Let me use the standard approach:
    # 3x3 Strassen: rank 23 for the outer structure
    # 2x2 Strassen: rank 7 for each 4x4 inner multiplication
    # Total rank = 23 * 7 = 161
    
    # Build using Kronecker product of coefficient tensors
    
    # 3x3 Strassen tensor T33:
    # For each of 7 terms, we have coefficients for A blocks and B blocks
    # A coefficients (which A_i to use):
    a_coeffs_33 = [
        [1, 1, 0],  # M1: A1+A2
        [1, 0, 0],  # M2: A1
        [1, 1, 1],  # M3: A1+A2+A3
        [0, 0, 1],  # M4: A3
        [0, 1, 0],  # M5: A2
        [0, 1, 1],  # M6: A2+A3
        [1, -1, 0], # M7: A1-A2
    ]
    b_coeffs_33 = [
        [1, 1, 0],  # M1: B1+B2
        [-1, 1, 0], # M2: B2-B1
        [0, 0, 1],  # M3: B3
        [0, -1, 1], # M4: B3-B2
        [0, 1, -1], # M5: B2-B3
        [-1, 0, 1], # M6: B3-B1
        [1, 1, 0],  # M7: B1+B2
    ]
    c_coeffs_33 = [
        [1, 0, 0],  # C1 from M1
        [1, 0, 0],  # C1 from M2 (with -1)
        [1, 0, 0],  # C1 from M3
        [1, 0, 0],  # C1 from M4 (with -1)
        [0, 1, 0],  # C2 from M1
        [0, 1, 0],  # C2 from M2 (with +1)
        [0, 1, 0],  # C2 from M4 (with +1)
        [0, 1, 0],  # C2 from M5 (with -1)
        [0, 0, 1],  # C3 from M3
        [0, 0, 1],  # C3 from M1 (with -1)
        [0, 0, 1],  # C3 from M5 (with +1)
        [0, 0, 1],  # C3 from M6 (with +1)
        [0, 0, 1],  # C3 from M7 (with -1)
    ]
    c_signs_33 = [
        1, -1, 1, -1,   # C1: M1 - M2 + M3 - M4
        1, 1, 1, -1,    # C2: M1 + M2 + M4 - M5
        1, -1, 1, 1, -1 # C3: M3 - M1 + M5 + M6 - M7
    ]
    
    # 2x2 Strassen tensor T22:
    a_coeffs_22 = [
        [1, 0],  # M1: A1
        [1, 0],  # M2: A1
        [0, 1],  # M3: A2
        [1, 0],  # M4: A1
    ]
    # Wait, 2x2 Strassen is:
    # M1 = A * (B1+B2) where A is the full 2x2 block
    # M2 = (A1-A2) * B2
    # M3 = (A1+A2) * B1  
    # M4 = A * (B2-B1)
    # This doesn't split A into sub-blocks the same way.
    
    # Let me use a cleaner 2x2 Strassen:
    # For multiplying (2x2)(2x2):
    # M1 = A11 * (B11 + B12)
    # M2 = (A11 + A12) * B21
    # M3 = (A12 - A11) * B11
    # M4 = A22 * (B12 + B22)
    # M5 = (A11 + A22) * (B11 + B22)
    # M6 = (A12 + A22) * (B22 - B11)
    # M7 = (A11 + A12) * (B12 - B21)
    # C11 = M5 + M4 - M6 - M7
    # C12 = M1 + M7
    # C21 = M2 + M3
    # C22 = M5 + M1 - M3 - M4
    
    # This is getting complex. Let me just use the schoolbook for the inner 4x4.
    # Total rank = 23 * 16 = 368, which is worse than 96.
    
    # Better: just use schoolbook with rank 96.
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
# EVOLVE-BLOCK-END
