"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    # Pad to 4x4x4x6
    n4, m4, p4 = 4, 4, 6
    rank = 49  # Strassen for 4x4x4x4
    
    U = [[0] * rank for _ in range(n4 * m4)]  # 16 x 49
    V = [[0] * rank for _ in range(m4 * p4)]  # 24 x 49
    W = [[0] * rank for _ in range(p4 * n4)]  # 24 x 49
    
    # Strassen for 2x2x2x2: rank 7
    # Define the 7 products for 2x2
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # For 4x4, we have 7 "Strassen steps", each involving 2x2 blocks.
    # Each 2x2 block multiplication uses the 7-product Strassen.
    
    # This is complex. Let me define the 49 scalar multiplications.
    
    # I'll index the 4x4 matrix as A[i][j] for i,j in 0..3
    # Similarly for B and C.
    
    # The 49 products are indexed by (s, t) where s in 0..6 (Strassen step), t in 0..6 (inner Strassen)
    
    # For each Strassen step s, we have 2x2 blocks.
    # Let me define the block structure.
    
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
    
    # Strassen:
    # M00 = (A00+A11)*(B00+B11)
    # M01 = (A10+A11)*B00
    # M02 = A00*(B01-B11)
    # M03 = A11*(B10-B00)
    # M04 = (A00+A01)*B11
    # M05 = (A10-A00)*(B00+B01)
    # M06 = (A01-A11)*(B10+B11)
    
    # C00 = M00 + M03 - M04 + M06
    # C01 = M02 + M04
    # C10 = M01 + M03
    # C11 = M00 - M01 + M02 + M05
    
    # Each Mij is a 2x2 matrix product, which uses 7 scalar multiplications.
    
    # So total 7*7 = 49 scalar multiplications.
    
    # Let me define the 49 terms.
    
    terms = []
    
    # For each of the 7 outer Strassen products, define the 7 inner Strassen products.
    
    # Outer Strassen product 0: M00 = (A00+A11)*(B00+B11)
    # A00+A11: A[0][0]+A[2][2], A[0][1]+A[2][3], A[1][0]+A[3][2], A[1][1]+A[3][3]
    # B00+B11: B[0][0]+B[2][0], B[0][1]+B[2][1], B[1][0]+B[3][0], B[1][1]+B[3][1]
    # Inner Strassen for 2x2:
    # M00_0 = (A00_00+A00_11)*(B00_00+B00_11)  # This is getting too nested
    
    # This is too complex to do by hand. Let me just use the schoolbook method.
    
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
