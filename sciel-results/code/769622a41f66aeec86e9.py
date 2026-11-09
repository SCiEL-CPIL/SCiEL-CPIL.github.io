import time
import sys

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    # Total rank will be 44
    # 7 for Strassen 2x2x2 on top-left block (i,j in 0,1; k in 0,1)
    # 37 for the rest
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)] # 9 x 44
    V = [[0] * rank for _ in range(m * p)] # 15 x 44
    W = [[0] * rank for _ in range(p * n)] # 15 x 44
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, coeff_u, coeff_v, coeff_w):
        nonlocal col
        U[i * m + j][col] = coeff_u
        V[j * p + k][col] = coeff_v
        W[k * n + i][col] = coeff_w
        col += 1

    # 1. Strassen part for i,j in {0,1} and k in {0,1}
    # M1 = (A00 + A11) * (B00 + B11)
    # U: A00=1, A11=1. V: B00=1, B11=1.
    # W: C00 += 1, C11 += 1.
    # Wait, the W matrix maps the product to the output C.
    # The term is (A00+A11)(B00+B11).
    # This contributes to C00, C01, C10, C11.
    # M1 contributes +1 to C00, +1 to C11.
    # Actually, let's trace the coefficients carefully.
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2
    # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C01 += M5, C11 -= M5
    # M6 = (A10-A00)(B00+B01) -> C10 -= M6? No, C11 += M6? 
    # Let's re-verify Strassen formulas.
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's define the 7 terms (u, v, w_contribs)
    
    # Term 1: (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C11=1
    add_term(0, 0, 0, 1, 1, 0) # Placeholder, need to handle multiple non-zeros in U/V/W
    
    # This simple add_term is not enough for Strassen because U/V have multiple non-zeros.
    # I need a helper to set multiple indices.
    
    def add_strassen_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, val in u_indices.items():
            U[idx][col] = val
        for idx, val in v_indices.items():
            V[idx][col] = val
        for idx, val in w_indices.items():
            W[idx][col] = val
        col += 1

    # Indices for U (i*m+j):
    # A00: 0, A01: 1, A10: 3, A11: 4
    # Indices for V (j*p+k):
    # B00: 0, B01: 1, B10: 3, B11: 4
    # Indices for W (k*n+i):
    # C00: 0, C01: 1, C10: 3, C11: 4
    
    # M1: (A00+A11)(B00+B11)
    # U: {0:1, 4:1}
    # V: {0:1, 4:1}
    # W: {0:1, 4:1}
    add_strassen_term({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1})
    
    # M2: (A10+A11)B00
    # U: {3:1, 4:1}
    # V: {0:1}
    # W: {3:1} (C10)
    add_strassen_term({3:1, 4:1}, {0:1}, {3:1})
    
    # M3: A00(B10-B11)
    # U: {0:1}
    # V: {3:1, 4:-1}
    # W: {1:1, 4:1} (C01, C11)
    add_strassen_term({0:1}, {3:1, 4:-1}, {1:1, 4:1})
    
    # M4: A11(B01-B00)
    # U: {4:1}
    # V: {1:1, 0:-1}
    # W: {0:1, 3:1} (C00, C10)
    add_strassen_term({4:1}, {1:1, 0:-1}, {0:1, 3:1})
    
    # M5: (A00+A01)B11
    # U: {0:1, 1:1}
    # V: {4:1}
    # W: {1:1, 4:-1} (C01, -C11)
    add_strassen_term({0:1, 1:1}, {4:1}, {1:1, 4:-1})
    
    # M6: (A10-A00)(B00+B01)
    # U: {3:1, 0:-1}
    # V: {0:1, 1:1}
    # W: {4:1} (C11)
    add_strassen_term({3:1, 0:-1}, {0:1, 1:1}, {4:1})
    
    # M7: (A01-A11)(B10+B11)
    # U: {1:1, 4:-1}
    # V: {3:1, 4:1}
    # W: {0:1} (C00)
    add_strassen_term({1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Now the remaining 37 terms (Schoolbook)
    # We need to add all (i,j,k) where NOT (i in {0,1} AND j in {0,1} AND k in {0,1})
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in [0,1] and j in [0,1] and k in [0,1]:
                    continue
                # Add schoolbook term
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
