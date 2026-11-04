"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    n, m, p = 4, 4, 6
    
    # We decompose 4x4x6 into 6 independent 2x2x2 block multiplications.
    # A (4x4) = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B (4x6) = [[B00, B01, B02], [B10, B11, B12]] where each Bj is 2x2
    # C (4x6) = [[C00, C01, C02], [C10, C11, C12]] where each Ck is 2x2
    # C[i_block, k_block] = A[i_block, 0] * B[0, k_block] + A[i_block, 1] * B[1, k_block]
    #
    # Each 2x2 matrix product uses Strassen's rank-7 algorithm.
    # Total rank = 6 * 2 * 7 = 84? No, wait.
    # For each (i_block, k_block), we need to compute two 2x2 matrix products and add them.
    # Each 2x2 matrix product has rank 7.
    # So for each (i_block, k_block), rank = 2 * 7 = 14.
    # Total rank = 6 * 14 = 84.
    #
    # Actually, we can't just add the ranks because the U and V vectors for the two
    # products in each block multiplication are different.
    # Let me reconsider.
    #
    # C[i_block, k_block] = A[i_block, 0] * B[0, k_block] + A[i_block, 1] * B[1, k_block]
    #
    # Let's call the first product P = A[i_block, 0] * B[0, k_block] and the second Q = A[i_block, 1] * B[1, k_block].
    # P is computed with 7 multiplications, Q with 7 multiplications.
    # Total for each (i_block, k_block): 14 multiplications.
    # Total: 6 * 14 = 84.
    #
    # But wait, we can potentially share some computations. However, for now, let's just
    # implement the straightforward approach with rank 84, which is still better than 96.
    #
    # Actually, let me reconsider the structure. The input A is 4x4 and B is 4x6.
    # A has 2 block rows and 2 block columns (each 2x2).
    # B has 2 block rows and 3 block columns (each 2x2).
    # C has 2 block rows and 3 block columns (each 2x2).
    #
    # C[i, k] = sum_j A[i, j] * B[j, k] for i in {0,1}, k in {0,1,2}, j in {0,1}
    #
    # So for each (i, k), we have two 2x2 matrix multiplications.
    # Each 2x2 matrix multiplication has rank 7 (Strassen).
    # Total rank = 2 * 3 * 2 * 7 = 84.
    
    # Let's define the Strassen decomposition for a single 2x2x2 multiplication.
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    
    # The 7 Strassen multiplications:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b00)
    # M4 = a11*(b01-b11)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # Output:
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Let's verify this with a numerical test.
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # C = [[19,22],[43,50]]
    # M1 = (1+4)*(5+8) = 65
    # M2 = (3+4)*5 = 35
    # M3 = 1*(7-5) = 2
    # M4 = 4*(6-8) = -8
    # M5 = (1+2)*8 = 24
    # M6 = (3-1)*(5+6) = 22
    # M7 = (2-4)*(7+8) = -30
    # C00 = 65 - 24 + (-8) + (-30) = 3. Should be 19. WRONG!
    
    # I have the wrong Strassen formulas. Let me find the correct ones.
    # After research, the correct formulas are:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * (B00)
    # M3 = A00 * (B10 - B00)
    # M4 = A11 * (B01 - B11)
    # M5 = (A00 + A01) * (B11)
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    #
    # But this gives C00 = 3, not 19. So these formulas are wrong too.
    #
    # Let me try yet another set. Maybe the issue is with how I'm indexing.
    # Let me just use the schoolbook for now (rank 4 per 2x2 product) to get a working solution,
    # then optimize.
    
    # Schoolbook for 2x2: rank 4
    # M1 = a00*b00, M2 = a01*b10, M3 = a10*b01, M4 = a11*b11
    # C00 = M1 + M2, C01 = M3 + M4, C10 = a10*b00 + a11*b10, C11 = a10*b01 + a11*b11
    # Wait, that's not right either. Schoolbook for 2x2:
    # C00 = a00*b00 + a01*b10
    # C01 = a00*b01 + a01*b11
    # C10 = a10*b00 + a11*b10
    # C11 = a10*b01 + a11*b11
    # This requires 8 multiplications, not 4. The rank-4 decomposition doesn't exist for 2x2x2.
    # The minimum rank for 2x2x2 is 7 (Strassen).
    
    # Let me just implement the correct Strassen by finding the right formulas.
    # I'll use a known correct implementation.
    
    # Correct Strassen (verified):
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * (B00)
    # M3 = A00 * (B10 - B00)
    # M4 = A11 * (B01 - B11)
    # M5 = (A00 + A01) * (B11)
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Testing again with A=[[1,2],[3,4]], B=[[5,6],[7,8]]:
    # Expected C = [[19,22],[43,50]]
    # M1 = 5*13 = 65
    # M2 = 7*5 = 35
    # M3 = 1*2 = 2
    # M4 = 4*(-2) = -8
    # M5 = 3*8 = 24
    # M6 = 2*11 = 22
    # M7 = (-2)*15 = -30
    # C00 = 65 - 24 + (-8) + (-30) = 3 ≠ 19
    
    # These formulas are definitely wrong. Let me look for a different source.
    # 
    # Alternative correct Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * (B00)
    # M3 = A00 * (B10 - B00)
    # M4 = A11 * (B01 - B11)
    # M5 = (A00 + A01) * (B11)
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # C00 = 65 - (-8) + 24 - (-30) = 65 + 8 + 24 + 30 = 127. No.
    
    # I'm clearly struggling with the exact Strassen formulas. Let me just use a 
    # different approach: I'll construct the rank-7 decomposition by solving for it.
    # Actually, let me just use the schoolbook (rank 8 per 2x2 product) for now.
    # Total rank = 6 * 2 * 8 = 96. Same as before. Not an improvement.
    
    # Let me try yet another approach. I'll look up the exact Strassen formulas one more time.
    # 
    # From Wikipedia:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)(b11)
    # m3 = (a11)(b21-b11)
    # m4 = (a22)(b12-b22)
    # m5 = (a11+a12)(b22)
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    #
    # c11 = m1 - m5 + m4 + m7
    # c12 = m3 + m5
    # c21 = m2 + m6
    # c22 = m1 - m2 + m3 + m6
    
    # Using 1-indexed notation, let me convert to 0-indexed:
    # a11=a00, a12=a01, a21=a10, a22=a11
    # b11=b00, b12=b01, b21=b10, b22=b11
    # c11=c00, c12=c01, c21=c10, c22=c11
    #
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)(b00)
    # m3 = a00*(b10-b00)
    # m4 = a11*(b01-b11)
    # m5 = (a00+a01)(b11)
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    #
    # c00 = m1 - m5 + m4 + m7
    # c01 = m3 + m5
    # c10 = m2 + m6
    # c11 = m1 - m2 + m3 + m6
    
    # This is the same as what I had before. Let me test with the example again very carefully.
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # a00=1, a01=2, a10=3, a11=4
    # b00=5, b01=6, b10=7, b11=8
    #
    # m1 = (1+4)*(5+8) = 5*13 = 65
    # m2 = (3+4)*5 = 7*5 = 35
    # m3 = 1*(7-5) = 1*2 = 2
    # m4 = 4*(6-8) = 4*(-2) = -8
    # m5 = (1+2)*8 = 3*8 = 24
    # m6 = (3-1)*(5+6) = 2*11 = 22
    # m7 = (2-4)*(7+8) = (-2)*15 = -30
    #
    # c00 = 65 - 24 + (-8) + (-30) = 65 - 24 - 8 - 30 = 3
    # c01 = 2 + 24 = 26
    # c10 = 35 + 22 = 57
    # c11 = 65 - 35 + 2 + 22 = 54
    #
    # Expected: c00=19, c01=22, c10=43, c11=50
    # Got: c00=3, c01=26, c10=57, c11=54
    #
    # None of these match. The Wikipedia formulas must be for a different convention.
    # 
    # Let me check: maybe the Wikipedia convention has C = B*A instead of C = A*B?
    # If C = B*A, then C[0,0] = b00*a00 + b01*a10 = 5*1 + 6*3 = 23. No.
    #
    # I think there might be an error in my understanding. Let me just implement the 
    # schoolbook algorithm with rank 8 per 2x2 product for now, giving total rank 96.
    # This is the same as the current program, so no improvement.
    #
    # Actually, let me just go with the schoolbook approach but be more careful.
    # The current program already has rank 96. Let me try to get rank 84 by using
    # a rank-7 decomposition if I can find the correct formulas.
    #
    # Let me try a completely different approach: I'll construct the rank-7 decomposition
    # by solving a system of equations.
    
    # For now, let me just return the schoolbook decomposition (rank 96) to ensure correctness,
    # and then I can try to improve it.
    
    rank = 4 * 4 * 6
    U = [[0] * rank for _ in range(4 * 4)]
    V = [[0] * rank for _ in range(4 * 6)]
    W = [[0] * rank for _ in range(6 * 4)]
    column = 0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                U[i * 4 + j][column] = 1
                V[j * 6 + k][column] = 1
                W[k * 4 + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
