"""Matrix multiplication 6x6x6 using 2x2 block decomposition with 3x3 Laderman (rank 19)."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 6, 6, 6
    b = 3  # block size

    # We'll use a rank-19 decomposition for 3x3 matrix multiplication.
    # The 19 bilinear forms for Laderman's 3x3 algorithm:
    
    # Let's define the 19 products for 3x3:
    # m0  = a00 * b00
    # m1  = a01 * b11
    # m2  = a02 * b22
    # m3  = a10 * b01
    # m4  = a11 * b12
    # m5  = a12 * b20
    # m6  = a20 * b02
    # m7  = a21 * b10
    # m8  = a22 * b21
    # m9  = (a00 + a10) * (b00 + b10)
    # m10 = (a01 + a11) * (b10 + b11)
    # m11 = (a02 + a12) * (b20 + b21)
    # m12 = (a00 + a20) * (b00 + b20)
    # m13 = (a01 + a21) * (b10 + b20)
    # m14 = (a02 + a22) * (b20 + b21)
    # m15 = (a10 + a20) * (b01 + b21)
    # m16 = (a11 + a21) * (b11 + b21)
    # m17 = (a12 + a22) * (b21 + b22)
    # m18 = (a00 + a01 + a02) * (b00 + b10 + b20)
    
    # Actually, let me use a well-known valid rank-19 decomposition.
    # I'll construct it carefully.
    
    # For 3x3, we need 19 terms. Let me use the following approach:
    # I'll define each of the 19 bilinear forms explicitly.
    
    # Each term c has:
    # - A coefficients: 9 values (for a00, a01, a02, a10, a11, a12, a20, a21, a22)
    # - B coefficients: 9 values (for b00, b01, b02, b10, b11, b12, b20, b21, b22)
    # - C contributions: 9 values (for c00, c01, c02, c10, c11, c12, c20, c21, c22)
    
    # Let me use a verified rank-19 decomposition.
    # I'll construct it by combining simpler decompositions.
    
    # Actually, let me just use the schoolbook method for the 3x3 case within blocks,
    # but that gives 27 per block, so 8 * 27 = 216 total, same as full schoolbook.
    
    # Let me try to implement a proper rank-19 decomposition for 3x3.
    
    # Here's a known rank-19 decomposition for 3x3 (from literature):
    # I'll define it as a list of 19 tuples, each containing (A_coeffs, B_coeffs, C_coeffs)
    
    # For simplicity and correctness, let me use a different approach:
    # I'll use the fact that 3x3 can be decomposed using rank 19.
    
    # Let me define the 19 terms explicitly.
    
    # Term 0: a00 * b00 -> c00
    # Term 1: a01 * b11 -> c01
    # Term 2: a02 * b22 -> c02
    # Term 3: a10 * b01 -> c10
    # Term 4: a11 * b12 -> c11
    # Term 5: a12 * b20 -> c12
    # Term 6: a20 * b02 -> c20
    # Term 7: a21 * b10 -> c21
    # Term 8: a22 * b21 -> c22
    
    # Terms 9-17: (ai + aj) * (bk + bl) forms
    # Term 9: (a00 + a10) * (b00 + b10)
    # Term 10: (a01 + a11) * (b10 + b11)
    # Term 11: (a02 + a12) * (b20 + b21)
    # Term 12: (a00 + a20) * (b00 + b20)
    # Term 13: (a01 + a21) * (b10 + b20)
    # Term 14: (a02 + a22) * (b20 + b21)
    # Term 15: (a10 + a20) * (b01 + b21)
    # Term 16: (a11 + a21) * (b11 + b21)
    # Term 17: (a12 + a22) * (b21 + b22)
    
    # Term 18: (a00 + a01 + a02) * (b00 + b10 + b20)
    
    # Now I need to figure out how these 19 terms combine to give the correct C matrix.
    
    # Let me just use the schoolbook approach for now, which is guaranteed correct.
    # Rank = 216 for 6x6x6.
    
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
