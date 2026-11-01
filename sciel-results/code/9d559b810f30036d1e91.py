"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen rank-7 decomposition for 2x2 matrix multiplication.
    # Output factor follows the official trace convention W[k*n+i, r].
    n, m, p = DIMENSIONS  # all 2
    
    # Strassen's 7 intermediate products for 2x2 matrix multiplication:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Result:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U matrix: 4 rows (A00, A01, A10, A11) x 7 cols (M1..M7)
    # Row 0: A00 coefficients
    # Row 1: A01 coefficients
    # Row 2: A10 coefficients
    # Row 3: A11 coefficients
    U = [
        [1,  0, 1,  0, 1,  0, 1],  # A00
        [0,  1, 0,  0, 1,  0, -1], # A01
        [0,  0, 0,  0, 0,  1, 0],  # A10
        [1,  0, 0,  1, 0,  0, 0],  # A11
        # Wait, let me re-derive:
        # M1 = (A00+A11)*(B00+B11): A00 coeff=1, A01=0, A10=0, A11=1
        # M2 = (A10+A11)*B00: A00=0, A01=0, A10=1, A11=1
        # M3 = A00*(B01-B11): A00=1, A01=0, A10=0, A11=0
        # M4 = A11*(B10-B00): A00=0, A01=0, A10=0, A11=1
        # M5 = (A00+A01)*B11: A00=1, A01=1, A10=0, A11=0
        # M6 = (A10-A00)*(B00+B01): A00=-1, A01=0, A10=1, A11=0
        # M7 = (A01-A11)*(B10+B11): A00=0, A01=1, A10=0, A11=-1
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ]
    # Correct U:
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ]
    
    # V matrix: 4 rows (B00, B01, B10, B11) x 7 cols
    # M1: B00+B11 -> B00=1, B01=0, B10=0, B11=1
    # M2: B00 -> B00=1, B01=0, B10=0, B11=0
    # M3: B01-B11 -> B00=0, B01=1, B10=0, B11=-1
    # M4: B10-B00 -> B00=-1, B01=0, B10=1, B11=0
    # M5: B11 -> B00=0, B01=0, B10=0, B11=1
    # M6: B00+B01 -> B00=1, B01=1, B10=0, B11=0
    # M7: B10+B11 -> B00=0, B01=0, B10=1, B11=1
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],    # B01
        [0, 0, 0, 1, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1],   # B11
    ]
    
    # W matrix: 4 rows (C00, C01, C10, C11) x 7 cols
    # Following trace convention W[k*n+i, r] where n=2
    # W[0, r] = C00 (k=0, i=0), W[1, r] = C10 (k=0, i=1), W[2, r] = C01 (k=1, i=0), W[3, r] = C11 (k=1, i=1)
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ]
    
    rank = 7
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
