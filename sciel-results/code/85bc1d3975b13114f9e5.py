"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 29 decomposition: One Strassen 2x2x2 block + Schoolbook for the rest.
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen uses 7 products. Schoolbook uses 8.
    # Remaining terms: 30 - 8 = 22.
    # Total rank = 7 + 22 = 29.
    
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen Block for C[0,0], C[0,1], C[1,0], C[1,1]
    # A indices: 0,1 (col 0), 2,3 (col 1) -> A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    # B indices: 0,1 (row 0, col 0), 2,3 (row 0, col 1), 4,5 (row 1, col 0), 6,7 (row 1, col 1)
    # Wait, B is 3x5.
    # B[0,0] is V[0*5+0] = V[0]
    # B[0,1] is V[0*5+1] = V[1]
    # B[1,0] is V[1*5+0] = V[5]
    # B[1,1] is V[1*5+1] = V[6]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0*2+0][col] = 1  # C00
    W[1*2+1][col] = 1  # C11
    col += 1
    
    # M2 = (A10+A11)B00
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[1*2+0][col] = 1  # C10
    col += 1
    
    # M3 = A00(B10+B11)
    U[0][col] = 1
    V[5][col] = 1; V[6][col] = 1
    W[0*2+1][col] = 1  # C01
    col += 1
    
    # M4 = A11(B00-B11)
    U[3][col] = 1
    V[0][col] = 1; V[6][col] = -1
    W[0*2+0][col] = 1  # C00
    W[1*2+0][col] = 1  # C10
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0*2+0][col] = -1 # C00 (subtract)
    W[0*2+1][col] = 1  # C01
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[1*2+1][col] = 1  # C11
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[3][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0*2+0][col] = 1  # C00
    W[1*2+1][col] = 1  # C11
    col += 1
    
    # Schoolbook for the remaining 22 terms
    # C02, C03, C04, C12, C13, C14
    # And C00, C01, C10, C11 are already handled by Strassen?
    # No, Strassen handles C00, C01, C10, C11.
    # We need to calculate C02, C03, C04, C12, C13, C14.
    # These depend on A[0,0..2], A[1,0..2] and B[0..2, 2..4].
    
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C03 = A00*B03 + A01*B13 + A02*B23
    # C04 = A00*B04 + A01*B14 + A02*B24
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C13 = A10*B03 + A11*B13 + A12*B23
    # C14 = A10*B04 + A11*B14 + A12*B24
    
    # Note: A02 is U[2], A12 is U[5].
    # B02 is V[2], B12 is V[7], B22 is V[12].
    # B03 is V[3], B13 is V[8], B23 is V[13].
    # B04 is V[4], B14 is V[9], B24 is V[14].
    
    # Term 8: A00*B02
    U[0][col] = 1; V[2][col] = 1; W[2*2+0][col] = 1; col += 1
    # Term 9: A01*B12
    U[1][col] = 1; V[7][col] = 1; W[2*2+0][col] = 1; col += 1
    # Term 10: A02*B22
    U[2][col] = 1; V[12][col] = 1; W[2*2+0][col] = 1; col += 1
    # Term 11: A10*B02
    U[3][col] = 1; V[2][col] = 1; W[2*2+1][col] = 1; col += 1
    # Term 12: A11*B12
    U[4][col] = 1; V[7][col] = 1; W[2*2+1][col] = 1; col += 1
    # Term 13: A12*B22
    U[5][col] = 1; V[12][col] = 1; W[2*2+1][col] = 1; col += 1
    
    # Term 14: A00*B03
    U[0][col] = 1; V[3][col] = 1; W[3*2+0][col] = 1; col += 1
    # Term 15: A01*B13
    U[1][col] = 1; V[8][col] = 1; W[3*2+0][col] = 1; col += 1
    # Term 16: A02*B23
    U[2][col] = 1; V[13][col] = 1; W[3*2+0][col] = 1; col += 1
    # Term 17: A10*B03
    U[3][col] = 1; V[3][col] = 1; W[3*2+1][col] = 1; col += 1
    # Term 18: A11*B13
    U[4][col] = 1; V[8][col] = 1; W[3*2+1][col] = 1; col += 1
    # Term 19: A12*B23
    U[5][col] = 1; V[13][col] = 1; W[3*2+1][col] = 1; col += 1
    
    # Term 20: A00*B04
    U[0][col] = 1; V[4][col] = 1; W[4*2+0][col] = 1; col += 1
    # Term 21: A01*B14
    U[1][col] = 1; V[9][col] = 1; W[4*2+0][col] = 1; col += 1
    # Term 22: A02*B24
    U[2][col] = 1; V[14][col] = 1; W[4*2+0][col] = 1; col += 1
    # Term 23: A10*B04
    U[3][col] = 1; V[4][col] = 1; W[4*2+1][col] = 1; col += 1
    # Term 24: A11*B14
    U[4][col] = 1; V[9][col] = 1; W[4*2+1][col] = 1; col += 1
    # Term 25: A12*B24
    U[5][col] = 1; V[14][col] = 1; W[4*2+1][col] = 1; col += 1
    
    # We have 7 Strassen + 18 Schoolbook = 25 terms?
    # Wait, I listed 18 schoolbook terms.
    # Total 25.
    # But I need 29.
    # Where are the other 4?
    # Ah, the Strassen block handles C00, C01, C10, C11.
    # The schoolbook part handles C02, C03, C04, C12, C13, C14.
    # That's 6 outputs.
    # Each output has 3 terms (since m=3).
    # So 6 * 3 = 18 terms.
    # Total 7 + 18 = 25.
    # Why did I think 29?
    # Schoolbook is 30.
    # Strassen saves 1 term.
    # So 29.
    # My count of schoolbook terms for the rest is 18.
    # 30 - 8 (Strassen block schoolbook) = 22.
    # I only listed 18.
    # I missed the terms for C00, C01, C10, C11 that are NOT part of the Strassen block?
    # No, Strassen block uses A00, A01, A10, A11 and B00, B01, B10, B11.
    # It calculates C00, C01, C10, C11 exactly.
    # So no other terms are needed for these outputs.
    # So the total rank is 25?
    # Let's re-verify.
    # C00 = A00B00 + A01B10 + A02B20.
    # Strassen calculates A00B00 + A01B10 using M1..M7.
    # It does NOT calculate A02B20.
    # So I need to add A02B20, A02B21, A12B20, A12B21.
    
    # Term 26: A02*B20
    U[2][col] = 1; V[10][col] = 1; W[0*2+0][col] = 1; col += 1
    # Term 27: A02*B21
    U[2][col] = 1; V[11][col] = 1; W[1*2+0][col] = 1; col += 1
    # Term 28: A12*B20
    U[5][col] = 1; V[10][col] = 1; W[0*2+1][col] = 1; col += 1
    # Term 29: A12*B21
    U[5][col] = 1; V[11][col] = 1; W[1*2+1][col] = 1; col += 1
    
    # Total 29 terms.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
