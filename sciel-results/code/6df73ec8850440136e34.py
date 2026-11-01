import time
import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # We'll implement a rank-28 solution using Strassen for each 2x2 block of B
    # B is 2x8, so we split into four 2x2 blocks
    # A is 2x2
    # C = A @ B is 2x8, split into four 2x2 blocks
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 - M4 - M5 + M7
    # C01 = M4 + M5
    # C10 = M2 - M3 - M6 - M7
    # C11 = M3 + M6
    
    # For the full 2x8 problem, we have 4 such blocks
    # Total rank = 4 * 7 = 28
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)  # (4, 28)
    V = np.zeros((m * p, rank), dtype=float)  # (16, 28)
    W = np.zeros((p * n, rank), dtype=float)  # (16, 28)
    
    # Define Strassen products
    # Each product has coefficients for A, B, and C
    strassen_products = [
        # M1 = (A00 + A11) * (B00 + B11)
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): -1}},  # contributes to C00 (+1), C11 (-1)
        # M2 = (A10 + A11) * B00
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},  # contributes to C10 (+1), C11 (-1)
        # M3 = A00 * (B01 - B11)
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},  # contributes to C01 (+1), C11 (+1)
        # M4 = A11 * (B10 - B00)
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (0,1): 1, (1,0): -1}},  # contributes to C00 (+1), C01 (+1), C10 (-1)
        # M5 = (A00 + A01) * B11
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},  # contributes to C00 (-1), C01 (+1)
        # M6 = (A10 - A00) * (B00 + B01)
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1, (1,0): 1}},  # contributes to C11 (+1), C10 (+1)
        # M7 = (A01 - A11) * (B10 + B11)
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,0): -1}},  # contributes to C00 (+1), C10 (-1)
    ]
    
    # For each block q (0 to 3), assign 7 Strassen products
    for q in range(p // 2):
        k1 = 2 * q  # First column of this block
        k2 = 2 * q + 1  # Second column of this block
        
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r  # Tensor index
            
            # Fill U (A coefficients)
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
            
            # Fill V (B coefficients)
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub  # Map local k to global k
                V[j * p + k_global, t] = coeff
            
            # Fill W (C coefficients)
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub  # Map local k to global k
                W[k_global * n + i, t] = coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
