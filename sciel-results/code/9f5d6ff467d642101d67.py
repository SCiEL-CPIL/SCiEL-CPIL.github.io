"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    
    # Strassen decomposition for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We split the 3x10 matrix into:
    # B = [B0; B1; B2] where B0, B1, B2 are 1x10 row vectors.
    # We treat the 2x2 part (A00, A10 vs B0, B1) using Strassen.
    # The remaining part is A01, A11 (2x1) times B2 (1x10).
    
    # Total rank = 7 (Strassen) + 2*10 (Schoolbook for 2x1x10) = 27.
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]  # 2*3 = 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 3*10 = 30 rows
    W = [[0] * rank for _ in range(p * n)]  # 10*2 = 20 rows
    
    # Indices:
    # A is 2x3: A[i, j] where i in 0..1, j in 0..2.
    # U row index: i*m + j = i*3 + j.
    # A00 -> U[0], A01 -> U[1], A02 -> U[2]
    # A10 -> U[3], A11 -> U[4], A12 -> U[5]
    
    # B is 3x10: B[j, k] where j in 0..2, k in 0..9.
    # V row index: j*p + k = j*10 + k.
    # B0k -> V[k], B1k -> V[10+k], B2k -> V[20+k]
    
    # C is 2x10: C[i, k] where i in 0..1, k in 0..9.
    # W row index: k*n + i = k*2 + i.
    # C0k -> W[2k], C1k -> W[2k+1]
    
    # Strassen part (first 7 columns of U, V, W)
    # Terms:
    # M1: A part: A00+A11, B part: B0+B1
    # M2: A part: A10+A11, B part: B0
    # M3: A part: A00, B part: B01-B11 (Note: B01 means B row 0, col 1? No, B01 is B[0,1]. But here B is 3x10.
    #      In the 2x2x2 case, B is 2x2. Here we map B[0,:] and B[1,:].
    #      So B01 -> B[0,1], B11 -> B[1,1].
    #      Wait, the "2x2" part refers to rows 0 and 1 of A, and rows 0 and 1 of B.
    #      So the "k" dimension is 10.
    #      M3_k = A00 * (B[0,k] - B[1,k])
    #      M4_k = A11 * (B[1,k] - B[0,k])
    #      M5_k = (A00+A01) * B[1,k]
    #      M6_k = (A10-A00) * (B[0,k] + B[1,k])
    #      M7_k = (A01-A11) * (B[1,k] + B[1,k])? No.
    #      Let's re-verify Strassen formulas for C = A * B (2x2 * 2x2).
    #      M1 = (A00+A11)(B00+B11)
    #      M2 = (A10+A11)B00
    #      M3 = A00(B01-B11)
    #      M4 = A11(B10-B00)
    #      M5 = (A00+A01)B11
    #      M6 = (A10-A00)(B00+B01)
    #      M7 = (A01-A11)(B10+B11)
    #      
    #      C00 = M1 + M4 - M5 + M7
    #      C01 = M3 + M5
    #      C10 = M2 + M4
    #      C11 = M1 - M2 + M3 + M6
    #      
    #      In our case:
    #      A is 2x2 (rows 0,1; cols 0,1 of the 2x3 matrix).
    #      B is 2x10 (rows 0,1 of the 3x10 matrix).
    #      C is 2x10 (rows 0,1 of the 2x10 matrix).
    #      
    #      For each k in 0..9:
    #      M1_k = (A00+A11) * (B[0,k]+B[1,k])
    #      M2_k = (A10+A11) * B[0,k]
    #      M3_k = A00 * (B[0,k]-B[1,k])
    #      M4_k = A11 * (B[1,k]-B[0,k])
    #      M5_k = (A00+A01) * B[1,k]
    #      M6_k = (A10-A00) * (B[0,k]+B[1,k])
    #      M7_k = (A01-A11) * (B[1,k]+B[1,k])? No, (B10+B11) in 2x2 context.
    #             In 2x10 context, B is 2x10. So B10 is B[1,0], B11 is B[1,1].
    #             But the formula is per-element.
    #             C[i,k] depends on M_r,k.
    #             So M7_k = (A01-A11) * (B[1,k]+B[1,k])? No.
    #             In 2x2, M7 = (A01-A11)(B10+B11).
    #             Here, for output column k, the relevant B entries are B[1,k] and ...?
    #             Actually, Strassen works element-wise on the matrix product.
    #             So for column k of C (from the 2x2 block), we use the k-th column of B_block.
    #             So M7_k = (A01-A11) * (B[1,k] + B[1,k])? No.
    #             B10 and B11 in the 2x2 case are elements (1,0) and (1,1).
    #             In the 2x10 case, the "block" is 2x10.
    #             So M7_k = (A01-A11) * (B[1,k] + B[1,k]) is wrong.
    #             It should be M7_k = (A01-A11) * (B[1,k] + B[1,k])?
    #             Let's check:
    #             C11 (in 2x2) = M1 - M2 + M3 + M6.
    #             C1,k (row 1, col k of C_block) = M1_k - M2_k + M3_k + M6_k.
    #             C0,k (row 0, col k of C_block) = M1_k + M4_k - M5_k + M7_k.
    #             
    #             So:
    #             M1_k = (A00+A11)(B[0,k]+B[1,k])
    #             M2_k = (A10+A11)B[0,k]
    #             M3_k = A00(B[0,k]-B[1,k])
    #             M4_k = A11(B[1,k]-B[0,k])
    #             M5_k = (A00+A01)B[1,k]
    #             M6_k = (A10-A00)(B[0,k]+B[1,k])
    #             M7_k = (A01-A11)(B[1,k]+B[1,k])? No.
    #             In 2x2, M7 = (A01-A11)(B10+B11).
    #             B10 is element (1,0), B11 is element (1,1).
    #             In 2x10, for column k, the "elements" are just B[1,k].
    #             So where does the "B10+B11" come from?
    #             Ah, the formula M7 = (A01-A11)(B10+B11) is for the *scalar* product.
    #             If B is a matrix, M7 is a matrix? No, M7 is a scalar in the standard notation?
    #             No, in tensor decomposition, M_r are scalars.
    #             So for each k, we have a scalar M_r,k.
    #             M7_k = (A01-A11) * (B[1,k] + B[1,k])?
    #             Let's test with 2x2.
    #             C01 = M3 + M5.
    #             C01 = A00*B01 + A01*B11.
    #             M3 + M5 = A00(B01-B11) + (A00+A01)B11 = A00*B01 - A00*B11 + A00*B11 + A01*B11 = A00*B01 + A01*B11. Correct.
    #             C00 = M1 + M4 - M5 + M7.
    #             C00 = A00*B00 + A01*B10.
    #             M1 + M4 - M5 + M7 = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    #             = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    #             = A00*B00 + A01*B10. Correct.
    #             
    #             So for 2x10:
    #             M7_k = (A01-A11) * (B[1,k] + B[1,k])?
    #             No, in the 2x2 case, B10 and B11 are different elements.
    #             In the 2x10 case, for column k, the "B10" and "B11" are... just B[1,k]?
    #             No, the indices in M7 are (1,0) and (1,1) of the 2x2 matrix B.
    #             In our 2x10 matrix, the "2x2" block is actually 2x10.
    #             So the indices are (1,k) and (1,k)?
    #             No, the formula is fixed.
    #             M7 = (A01-A11)(B10+B11).
    #             If B is 2x10, then B10 is B[1,0] and B11 is B[1,1].
    #             But we want M7_k for each k.
    #             So M7_k = (A01-A11)(B[1,k] + B[1,k])?
    #             This implies B[1,k] + B[1,k] = 2*B[1,k].
    #             Let's check C00 again.
    #             C0,k = A00*B[0,k] + A01*B[1,k].
    #             M1_k + M4_k - M5_k + M7_k
    #             = (A00+A11)(B[0,k]+B[1,k]) + A11(B[1,k]-B[0,k]) - (A00+A01)B[1,k] + (A01-A11)(2*B[1,k])
    #             = A00*B[0,k] + A00*B[1,k] + A11*B[0,k] + A11*B[1,k] + A11*B[1,k] - A11*B[0,k] - A00*B[1,k] - A01*B[1,k] + 2*A01*B[1,k] - 2*A11*B[1,k]
    #             = A00*B[0,k] + A01*B[1,k]. Correct.
    #             
    #             So M7_k = (A01-A11) * 2 * B[1,k].
    
    # Let's construct the 7 Strassen terms.
    for r in range(7):
        # U (A part)
        if r == 0: # M1: A00+A11
            U[0][r] = 1; U[4][r] = 1
            # V (B part): B0+B1
            for k in range(10):
                V[k][r] = 1
                V[10+k][r] = 1
            # W (C part): C0 += M1, C1 += M1
            for k in range(10):
                W[2*k][r] = 1
                W[2*k+1][r] = 1
        elif r == 1: # M2: A10+A11
            U[3][r] = 1; U[4][r] = 1
            for k in range(10):
                V[k][r] = 1
            for k in range(10):
                W[2*k+1][r] = -1
        elif r == 2: # M3: A00
            U[0][r] = 1
            for k in range(10):
                V[k][r] = 1
                V[10+k][r] = -1
            for k in range(10):
                W[2*k][r] = 1
                W[2*k+1][r] = 1
        elif r == 3: # M4: A11
            U[4][r] = 1
            for k in range(10):
                V[k][r] = -1
                V[10+k][r] = 1
            for k in range(10):
                W[2*k][r] = 1
                W[2*k+1][r] = 1
        elif r == 4: # M5: A00+A01
            U[0][r] = 1; U[1][r] = 1
            for k in range(10):
                V[10+k][r] = 1
            for k in range(10):
                W[2*k][r] = -1
                W[2*k+1][r] = 1
        elif r == 5: # M6: A10-A00
            U[3][r] = 1; U[0][r] = -1
            for k in range(10):
                V[k][r] = 1
                V[10+k][r] = 1
            for k in range(10):
                W[2*k+1][r] = 1
        elif r == 6: # M7: A01-A11
            U[1][r] = 1; U[4][r] = -1
            for k in range(10):
                V[10+k][r] = 2
            for k in range(10):
                W[2*k][r] = 1
    
    # Schoolbook part for A[:, 2] * B[2, :]
    # A02, A12 (indices 2, 5 in U)
    # B[2, k] (indices 20+k in V)
    # C[0, k] += A02 * B[2, k]
    # C[1, k] += A12 * B[2, k]
    
    col = 7
    for i in range(2):
        for k in range(10):
            U[i*3 + 2][col] = 1  # A[i, 2]
            V[2*10 + k][col] = 1 # B[2, k]
            W[k*2 + i][col] = 1  # C[i, k]
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
