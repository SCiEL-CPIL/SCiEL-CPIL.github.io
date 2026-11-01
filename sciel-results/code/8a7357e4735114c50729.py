import numpy as np

def construct():
    n, m, p = 2, 3, 4
    # We aim for rank 22: use Strassen for the first 2 columns of B (2x2x2 block)
    # and schoolbook for the last 2 columns (2x1x2 each, total 4)
    # Wait, that gives 7 + 4 = 11 per row of A? No.
    # Let's decompose <2,3,4> as <2,2,4> + <2,1,4>
    # <2,2,4> can be done with 2 * <2,2,2> = 2 * 7 = 14 (using Strassen)
    # <2,1,4> is just 2*1*4 = 8 (schoolbook)
    # Total rank = 14 + 8 = 22.
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 12 x 22
    W = [[0] * rank for _ in range(p * n)]  # 8 x 22
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+b)(e+g)
    # M2 = (a+b)(h-f)
    # M3 = (c+d)(e)
    # M4 = d(h-e)
    # M5 = (a+d)(e+h)
    # M6 = (c-a)(f+h)
    # M7 = (b-d)(g+f)
    # C11 = M5 + M4 - M2 - M7
    # C12 = M1 + M2
    # C21 = M3 + M4
    # C22 = M1 - M3 + M5 + M6
    
    # We are multiplying A (2x3) by B (3x4).
    # Let A = [A1 | A2] where A1 is 2x2 (cols 0,1), A2 is 2x1 (col 2)
    # Let B = [B1; B2] where B1 is 2x4 (rows 0,1), B2 is 1x4 (row 2)
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2x2 * 2x4), we split B1 into two 2x2 blocks: B1a (cols 0,1) and B1b (cols 2,3)
    # C_left = A1*B1a (2x2)
    # C_right = A1*B1b (2x2)
    # We can compute C_left using Strassen with A1 and B1a.
    # We can compute C_right using Strassen with A1 and B1b.
    # This uses 7 + 7 = 14 multiplications.
    
    # For A2*B2 (2x1 * 1x4), this is just outer product, 8 multiplications.
    # Total 22.
    
    # Indices:
    # U rows: 0..5 correspond to A elements:
    # 0: A[0,0], 1: A[0,1], 2: A[0,2]
    # 3: A[1,0], 4: A[1,1], 5: A[1,2]
    
    # V rows: 0..11 correspond to B elements:
    # 0: B[0,0], 1: B[0,1], 2: B[0,2], 3: B[0,3]
    # 4: B[1,0], 5: B[1,1], 6: B[1,2], 7: B[1,3]
    # 8: B[2,0], 9: B[2,1], 10: B[2,2], 11: B[2,3]
    
    # W rows: 0..7 correspond to C elements (trace convention W[k*n+i]):
    # 0: C[0,0] (k=0,i=0), 1: C[0,1] (k=1,i=0)
    # 2: C[1,0] (k=0,i=1), 3: C[1,1] (k=1,i=1)
    # 4: C[2,0] (k=2,i=0), 5: C[2,1] (k=3,i=0)
    # 6: C[3,0] (k=2,i=1), 7: C[3,1] (k=3,i=1)
    
    # --- Strassen Block 1 (A1 * B1a) -> C[0:2, 0:2] ---
    # A1 = [[A00, A01], [A10, A11]] = [[U0, U1], [U3, U4]]
    # B1a = [[B00, B01], [B10, B11]] = [[V0, V1], [V4, V5]]
    
    # M1 = (A00+A10)(B00+B10)
    # M2 = (A00+A10)(B11-B01)
    # M3 = (A10+A11)B00
    # M4 = A11(B11-B00)
    # M5 = (A00+A11)(B00+B11)
    # M6 = (A10-A00)(B01+B11)
    # M7 = (A01-A11)(B10+B01)
    
    # C00 = M5 + M4 - M2 - M7
    # C01 = M1 + M2
    # C10 = M3 + M4
    # C11 = M1 - M3 + M5 + M6
    
    # Col 0: M1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[1][0] += 1   # C01
    W[3][0] += 1   # C11
    
    # Col 1: M2
    U[0][1] += 1; U[3][1] += 1
    V[5][1] += 1; V[1][1] += -1
    W[1][1] += 1   # C01
    W[0][1] += -1  # C00
    
    # Col 2: M3
    U[3][2] += 1; U[4][2] += 1
    V[0][2] += 1
    W[2][2] += 1   # C10
    
    # Col 3: M4
    U[4][3] += 1
    V[5][3] += 1; V[0][3] += -1
    W[2][3] += 1   # C10
    W[0][3] += 1   # C00
    
    # Col 4: M5
    U[0][4] += 1; U[4][4] += 1
    V[0][4] += 1; V[5][4] += 1
    W[0][4] += 1   # C00
    W[3][4] += 1   # C11
    
    # Col 5: M6
    U[3][5] += 1; U[0][5] += -1
    V[1][5] += 1; V[5][5] += 1
    W[3][5] += 1   # C11
    
    # Col 6: M7
    U[1][6] += 1; U[4][6] += -1
    V[4][6] += 1; V[1][6] += 1
    W[0][6] += -1  # C00
    W[3][6] += -1  # C11
    
    # --- Strassen Block 2 (A1 * B1b) -> C[0:2, 2:4] ---
    # B1b = [[B02, B03], [B12, B13]] = [[V2, V3], [V6, V7]]
    
    # Col 7: M1'
    U[0][7] += 1; U[3][7] += 1
    V[2][7] += 1; V[6][7] += 1
    W[5][7] += 1   # C02 (index 4) -> wait, W[k*n+i]. C[0,2] is k=2, i=0 -> W[2*2+0] = W[4]
    # Let's re-verify W indices.
    # W[k*n+i]
    # C[0,0] -> k=0, i=0 -> W[0]
    # C[0,1] -> k=1, i=0 -> W[1]
    # C[1,0] -> k=0, i=1 -> W[2]
    # C[1,1] -> k=1, i=1 -> W[3]
    # C[0,2] -> k=2, i=0 -> W[4]
    # C[0,3] -> k=3, i=0 -> W[5]
    # C[1,2] -> k=2, i=1 -> W[6]
    # C[1,3] -> k=3, i=1 -> W[7]
    
    # Correcting W indices for Block 1:
    # C00 -> W[0], C01 -> W[1], C10 -> W[2], C11 -> W[3]
    # My previous assignments:
    # Col 0 (M1): W[1], W[3] -> C01, C11. Correct.
    # Col 1 (M2): W[1], W[0] -> C01, C00. Correct.
    # Col 2 (M3): W[2] -> C10. Correct.
    # Col 3 (M4): W[2], W[0] -> C10, C00. Correct.
    # Col 4 (M5): W[0], W[3] -> C00, C11. Correct.
    # Col 5 (M6): W[3] -> C11. Correct.
    # Col 6 (M7): W[0], W[3] -> C00, C11. Correct.
    
    # Now Block 2:
    # C02 -> W[4], C03 -> W[5], C12 -> W[6], C13 -> W[7]
    
    # Col 7: M1' = (A00+A10)(B02+B12)
    U[0][7] += 1; U[3][7] += 1
    V[2][7] += 1; V[6][7] += 1
    W[5][7] += 1   # C03
    W[7][7] += 1   # C13
    
    # Col 8: M2' = (A00+A10)(B13-B03)
    U[0][8] += 1; U[3][8] += 1
    V[7][8] += 1; V[3][8] += -1
    W[5][8] += 1   # C03
    W[4][8] += -1  # C02
    
    # Col 9: M3' = (A10+A11)B02
    U[3][9] += 1; U[4][9] += 1
    V[2][9] += 1
    W[6][9] += 1   # C12
    
    # Col 10: M4' = A11(B13-B02)
    U[4][10] += 1
    V[7][10] += 1; V[2][10] += -1
    W[6][10] += 1  # C12
    W[4][10] += 1  # C02
    
    # Col 11: M5' = (A00+A11)(B02+B13)
    U[0][11] += 1; U[4][11] += 1
    V[2][11] += 1; V[7][11] += 1
    W[4][11] += 1  # C02
    W[7][11] += 1  # C13
    
    # Col 12: M6' = (A10-A00)(B03+B13)
    U[3][12] += 1; U[0][12] += -1
    V[3][12] += 1; V[7][12] += 1
    W[7][12] += 1  # C13
    
    # Col 13: M7' = (A01-A11)(B12+B03)
    U[1][13] += 1; U[4][13] += -1
    V[6][13] += 1; V[3][13] += 1
    W[4][13] += -1 # C02
    W[7][13] += -1 # C13
    
    # --- Schoolbook Block 3 (A2 * B2) -> C[0:2, 0:4] ---
    # A2 = [A02, A12] = [U2, U5]
    # B2 = [B20, B21, B22, B23] = [V8, V9, V10, V11]
    
    # C00 += A02*B20
    # C01 += A02*B21
    # C02 += A02*B22
    # C03 += A02*B23
    # C10 += A12*B20
    # C11 += A12*B21
    # C12 += A12*B22
    # C13 += A12*B23
    
    # Col 14: A02*B20
    U[2][14] += 1
    V[8][14] += 1
    W[0][14] += 1  # C00
    
    # Col 15: A02*B21
    U[2][15] += 1
    V[9][15] += 1
    W[1][15] += 1  # C01
    
    # Col 16: A02*B22
    U[2][16] += 1
    V[10][16] += 1
    W[4][16] += 1  # C02
    
    # Col 17: A02*B23
    U[2][17] += 1
    V[11][17] += 1
    W[5][17] += 1  # C03
    
    # Col 18: A12*B20
    U[5][18] += 1
    V[8][18] += 1
    W[2][18] += 1  # C10
    
    # Col 19: A12*B21
    U[5][19] += 1
    V[9][19] += 1
    W[3][19] += 1  # C11
    
    # Col 20: A12*B22
    U[5][20] += 1
    V[10][20] += 1
    W[6][20] += 1  # C12
    
    # Col 21: A12*B23
    U[5][21] += 1
    V[11][21] += 1
    W[7][21] += 1  # C13
    
    return {"U": U, "V": V, "W": W}
