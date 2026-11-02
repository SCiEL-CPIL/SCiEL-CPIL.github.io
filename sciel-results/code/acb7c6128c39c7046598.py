"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Matrix multiplication 2x2 * 2x12.
    We split the 2x12 matrix into 6 blocks of 2x2.
    Apply Strassen's algorithm to each 2x2 block multiplication.
    This reduces the rank from 48 (schoolbook) to 42 (6 * 7).
    """
    n, m, p = DIMENSIONS  # 2, 2, 12
    
    # Strassen decomposition for 2x2 * 2x2
    # Input A (2x2), B (2x2)
    # Output C (2x2)
    #
    # Strassen's 7 products:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each of the 6 blocks (columns 0-1, 2-3, ..., 10-11 of B):
    # The total rank is 6 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen coefficients for the 7 products
    # Each product is of form (linear combo of A) * (linear combo of B)
    # A is 2x2: indices (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B is 2x2: indices (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    
    # For product t, we have:
    # U[i, t] = coefficient of A[i] in the A-part
    # V[j, t] = coefficient of B[j] in the B-part
    # W[k, t] = coefficient in the output C
    
    # Strassen products (a_part, b_part, c_contrib)
    # a_part: dict from A-index to coeff
    # b_part: dict from B-index to coeff
    # c_contrib: dict from C-index to coeff
    
    strassen_products = [
        # M1 = (A11 + A22) * (B11 + B22)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A21 + A22) * B11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A11 * (B12 - B22)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A22 * (B21 - B11)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A11 + A12) * B22
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A21 - A11) * (B11 + B12)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A12 - A22) * (B21 + B22)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for block in range(6):
        # B block columns: 2*block, 2*block+1
        # B indices: (0, 2*block)=B11, (0, 2*block+1)=B12, (1, 2*block)=B21, (1, 2*block+1)=B22
        for t in range(7):
            col = block * 7 + t
            a_part, b_part, c_contrib = strassen_products[t]
            
            # Fill U: A has shape 2x2, flattened row-major: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
            for (ai, aj), coeff in a_part.items():
                U[ai * m + aj][col] = coeff
            
            # Fill V: B block has shape 2x2, flattened: (0,0)->2*block, (0,1)->2*block+1, (1,0)->2*block+2, (1,1)->2*block+3
            for (bi, bj), coeff in b_part.items():
                V[bi * p + 2*block + bj][col] = coeff
            
            # Fill W: C has shape 2x2, output index W[k*n+i] where k is col in B-block, i is row in A
            # C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 in local block coords
            # Global W index: for C element (ci, cj) where ci is row (0 or 1), cj is col in block (0 or 1)
            # W index = cj * n + ci = cj * 2 + ci
            for (ci, cj), coeff in c_contrib.items():
                W[cj * n + ci][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
