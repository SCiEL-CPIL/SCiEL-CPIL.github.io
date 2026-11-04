"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 * 2x2
    # Returns lists of coefficients for A, B, C
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    def get_strassen_terms():
        terms = []
        # M1 = (A00+A11)(B00+B11)
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0])) # C00 += M1
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1])) # C11 += M1
        
        # M2 = (A10+A11)B00
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0])) # C10 += M2
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, -1])) # C11 -= M2
        
        # M3 = A00(B01-B11)
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0])) # C01 += M3
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 0, 1])) # C11 += M3
        
        # M4 = A11(B10-B00)
        terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0])) # C00 += M4
        terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 0])) # C11 += M4 (Wait, C11 = M1 - M2 + M3 + M6. M4 is in C00 only? No.)
        
        # Let's re-verify Strassen formulas
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M7
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # I will generate the terms by iterating M1..M7 and adding to C00..C11 accordingly.
        
        return None # Placeholder

    # I will implement the addition directly in the loop for clarity.
    
    def add_strassen_block(r_offset, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        # a_row_offset, a_col_offset: top-left of A block (2x2)
        # b_row_offset, b_col_offset: top-left of B block (2x2)
        # c_row_offset, c_col_offset: top-left of C block (2x2)
        
        # M1
        # A: A00+A11 -> indices (0,0) and (1,1) relative to block
        # B: B00+B11 -> indices (0,0) and (1,1) relative to block
        # C: C00 += 1, C11 += 1
        
        # M2
        # A: A10+A11 -> indices (1,0) and (1,1)
        # B: B00 -> index (0,0)
        # C: C10 += 1, C11 -= 1
        
        # M3
        # A: A00 -> index (0,0)
        # B: B01-B11 -> indices (0,1) and (1,1)
        # C: C01 += 1, C11 += 1
        
        # M4
        # A: A11 -> index (1,1)
        # B: B10-B00 -> indices (1,0) and (0,0)
        # C: C00 += 1
        
        # M5
        # A: A00+A01 -> indices (0,0) and (0,1)
        # B: B11 -> index (1,1)
        # C: C00 -= 1, C01 += 1
        
        # M6
        # A: A10-A00 -> indices (1,0) and (0,0)
        # B: B00+B01 -> indices (0,0) and (0,1)
        # C: C11 += 1
        
        # M7
        # A: A01-A11 -> indices (0,1) and (1,1)
        # B: B10+B11 -> indices (1,0) and (1,1)
        # C: C00 -= 1, C10 += 1
        
        # Define the 7 terms as (A_coeffs, B_coeffs, C_coeffs)
        # Coeffs are dicts or lists of (local_idx, coeff)
        
        terms = [
            # M1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            # M3
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            # M4
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1}),
            # M5
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # M7
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,0):1})
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in terms:
            r = r_offset
            for (lr, lc), val in a_coeffs.items():
                gi = a_row_offset + lr
                gj = a_col_offset + lc
                U[gi * m + gj][r] += val
            
            for (lr, lc), val in b_coeffs.items():
                gj = b_row_offset + lr
                gk = b_col_offset + lc
                V[gj * p + gk][r] += val
            
            for (lr, lc), val in c_coeffs.items():
                gi = c_row_offset + lr
                gk = c_col_offset + lc
                W[gk * n + gi][r] += val
            
            r_offset += 1
        
        return r_offset

    # Part 1: A[0:2, :] * B[:, :]
    # Split B into B[:, 0:2] and B[:, 2:4]
    # Split A[0:2, :] into A[0:2, 0:2] and A[0:2, 2:4]
    
    # C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
    
    r = 0
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    r = add_strassen_block(r, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    r = add_strassen_block(r, 0, 2, 2, 0, 0, 0)
    
    # C[0:2, 2:4] = A[0:2, 0:2] * B[0:2, 2:4] + A[0:2, 2:4] * B[2:4, 2:4]
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    r = add_strassen_block(r, 0, 0, 0, 2, 0, 2)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    r = add_strassen_block(r, 0, 2, 2, 2, 0, 2)
    
    # Part 2: A[2:3, :] * B[:, :]
    # Schoolbook for 1x4 * 4x4
    for j in range(4):
        for k in range(4):
            U[2 * m + j][r] = 1.0
            V[j * p + k][r] = 1.0
            W[k * n + 2][r] = 1.0
            r += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
