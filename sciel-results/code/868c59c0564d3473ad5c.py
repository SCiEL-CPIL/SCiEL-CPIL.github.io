import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # A: n x m (2x2)
    # B: m x p (2x10)
    # C: n x p (2x10) -> Index k*n + i
    
    # We use Strassen's algorithm for 2x2 matrix multiplication.
    # We view B as a 2x2 matrix over the ring R = F^5 (vectors of length 5).
    # B is split into two blocks of 5 columns: B_L (cols 0-4) and B_R (cols 5-9).
    # B[0, 0] in R is B[0, 0:5]
    # B[0, 1] in R is B[0, 5:10]
    # B[1, 0] in R is B[1, 0:5]
    # B[1, 1] in R is B[1, 5:10]
    
    # A is a 2x2 matrix over F.
    
    # Strassen products (7 terms):
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Note: C00, C01, C10, C11 are vectors in R (length 5).
    # C[0, 0:5] corresponds to C00
    # C[0, 5:10] corresponds to C01
    # C[1, 0:5] corresponds to C10
    # C[1, 5:10] corresponds to C11
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)] # 4 x 35
    V = [[0] * rank for _ in range(m * p)] # 20 x 35
    W = [[0] * rank for _ in range(p * n)] # 20 x 35
    
    # Define Strassen terms
    # Each term is: (A_coeffs, B_coeffs, W_coeffs)
    # A_coeffs: list of (index, coeff) for A (indices 0,1,2,3 for 00,01,10,11)
    # B_coeffs: list of (row, col_block, coeff) for B (col_block 0 or 1)
    # W_coeffs: list of (c_row, c_col_block, coeff) for C (c_row 0 or 1, c_col_block 0 or 1)
    
    # A indices:
    # 0: A00, 1: A01, 2: A10, 3: A11
    
    # B indices:
    # row 0, block 0 -> B[0, 0:5]
    # row 0, block 1 -> B[0, 5:10]
    # row 1, block 0 -> B[1, 0:5]
    # row 1, block 1 -> B[1, 5:10]
    
    # C indices:
    # row 0, block 0 -> C[0, 0:5]
    # row 0, block 1 -> C[0, 5:10]
    # row 1, block 0 -> C[1, 0:5]
    # row 1, block 1 -> C[1, 5:10]
    
    strassen_terms = [
        {
            'A': [(0, 1), (3, 1)], # A00 + A11
            'B': [(0, 0, 1), (1, 1, 1)], # B00 + B11
            'W': [(0, 0, 1), (1, 1, 1)] # C00 += 1, C11 += 1
        },
        {
            'A': [(2, 1), (3, 1)], # A10 + A11
            'B': [(0, 0, 1)], # B00
            'W': [(1, 0, 1), (1, 1, -1)] # C10 += 1, C11 -= 1
        },
        {
            'A': [(0, 1)], # A00
            'B': [(0, 1, 1), (1, 1, -1)], # B01 - B11
            'W': [(0, 1, 1), (1, 1, 1)] # C01 += 1, C11 += 1
        },
        {
            'A': [(3, 1)], # A11
            'B': [(1, 0, 1), (0, 0, -1)], # B10 - B00
            'W': [(0, 0, 1), (1, 0, 1)] # C00 += 1, C10 += 1
        },
        {
            'A': [(0, 1), (1, 1)], # A00 + A01
            'B': [(1, 1, 1)], # B11
            'W': [(0, 0, -1), (0, 1, 1)] # C00 -= 1, C01 += 1
        },
        {
            'A': [(2, 1), (0, -1)], # A10 - A00
            'B': [(0, 0, 1), (0, 1, 1)], # B00 + B01
            'W': [(1, 1, 1)] # C11 += 1
        },
        {
            'A': [(1, 1), (3, -1)], # A01 - A11
            'B': [(1, 0, 1), (1, 1, 1)], # B10 + B11
            'W': [(0, 0, 1)] # C00 += 1
        }
    ]
    
    col = 0
    for term in strassen_terms:
        # For each of the 5 components in the ring R=F^5
        for t in range(5):
            # Fill U
            for a_idx, a_coeff in term['A']:
                U[a_idx][col] = a_coeff
            
            # Fill V
            for b_row, b_block, b_coeff in term['B']:
                b_col = b_block * 5 + t
                V[b_row * p + b_col][col] = b_coeff
            
            # Fill W
            for c_row, c_block, c_coeff in term['W']:
                c_col = c_block * 5 + t
                # C index is c_col * n + c_row
                W[c_col * n + c_row][col] = c_coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
