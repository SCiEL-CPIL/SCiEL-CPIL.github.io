"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 22 decomposition for 2x3x4
    # Strategy: 
    # Split k into 2+2.
    # Each part is 2x3x2.
    # Rank of 2x3x2 is 11.
    # Total rank 22.
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x3x2 block (k offset)
    def add_block(k_offset):
        nonlocal col
        # Rank 11 decomposition for 2x3x2
        # Based on Winograd/Strassen hybrid
        # A is 2x3, B is 3x2, C is 2x2
        # Indices for B: j*p + (k_offset + k)
        # Indices for C: (k_offset + k)*n + i
        
        # 11 multiplications
        # m1 = A00 * (B00 + B10)
        # m2 = A01 * (B01 + B11)
        # m3 = (A00 + A01) * (B00 + B01)
        # m4 = A10 * (B00 + B10)
        # m5 = A11 * (B01 + B11)
        # m6 = (A10 + A11) * (B00 + B01)
        # m7 = (A00 - A10) * B00
        # m8 = (A01 - A11) * B01
        # m9 = (A00 + A10) * B10
        # m10 = (A01 + A11) * B11
        # m11 = A02 * B20 + A12 * B21 ? No, this is 2x3x2, so j=0,1,2.
        
        # Actually, 2x3x2 rank is 11.
        # Let's use a known decomposition.
        # A = [[a00, a01, a02], [a10, a11, a12]]
        # B = [[b00, b01], [b10, b11], [b20, b21]]
        
        # Terms:
        # 1. a00 * (b00 + b10)
        # 2. a01 * (b01 + b11)
        # 3. (a00 + a01) * (b00 + b01)
        # 4. a10 * (b00 + b10)
        # 5. a11 * (b01 + b11)
        # 6. (a10 + a11) * (b00 + b01)
        # 7. (a00 - a10) * b00
        # 8. (a01 - a11) * b01
        # 9. (a00 + a10) * b10
        # 10. (a01 + a11) * b11
        # 11. (a02 * b20 + a12 * b21) ? No, this is 2 multiplications.
        
        # This simple list doesn't sum to 11 easily with the 3rd column.
        # Let's just use schoolbook for the 2x3x2 blocks?
        # Schoolbook 2x3x2 is 12.
        # 12 + 12 = 24. No improvement.
        
        # Let's use the rank 22 decomposition from the inspiration.
        # It likely uses Strassen on 2x2x2 and outer product on the rest.
        
        # For 2x3x4:
        # Split j into 2+1.
        # Part 1: 2x2x4. Rank 15?
        # Part 2: 2x1x4. Rank 8.
        # Total 23.
        
        # Split k into 2+2.
        # Part 1: 2x3x2. Rank 11.
        # Part 2: 2x3x2. Rank 11.
        # Total 22.
        
        # I need a rank 11 decomposition for 2x3x2.
        # A (2x3) * B (3x2) = C (2x2).
        
        # Known rank 11 for 2x3x2:
        # m1 = (a00+a01) * (b00+b01)
        # m2 = a00 * (b00+b10)
        # m3 = a01 * (b01+b11)
        # m4 = (a10+a11) * (b00+b01)
        # m5 = a10 * (b00+b10)
        # m6 = a11 * (b01+b11)
        # m7 = (a00-a10) * b00
        # m8 = (a01-a11) * b01
        # m9 = (a00+a10) * b10
        # m10 = (a01+a11) * b11
        # m11 = a02*b20 + a12*b21 ? No, this is 2 terms.
        
        # Actually, the 3rd column of A and B can be handled with 2 multiplications?
        # a02*b20 and a12*b21.
        # That's 2.
        # 10 + 2 = 12.
        
        # So rank 11 is hard to construct simply.
        # I will return the schoolbook rank 24 to be safe, or rank 22 if I can find it.
        # Given the constraints, I will return rank 24.
        
        return

    # Since I cannot guarantee a rank < 24 decomposition without exact coefficients,
    # and the inspiration mentioned 22, I will attempt to construct rank 22 
    # using a known 2x3x2 rank 11 decomposition if available, 
    # otherwise fall back to schoolbook.
    
    # For now, return schoolbook to ensure validity.
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
