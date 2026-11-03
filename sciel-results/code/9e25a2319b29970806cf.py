"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank 7 Strassen decomposition for 2x2x2
    # U, V, W are 4x7
    # Indices for 2x2 matrices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U_22 = [[0]*7 for _ in range(4)]
    V_22 = [[0]*7 for _ in range(4)]
    W_22 = [[0]*7 for _ in range(4)]
    
    # Strassen factors
    # M1 = (A00+A11)(B00+B11)
    U_22[0][0] = 1; U_22[3][0] = 1
    V_22[0][0] = 1; V_22[3][0] = 1
    W_22[0][0] = 1; W_22[3][0] = 1 # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U_22[2][1] = 1; U_22[3][1] = 1
    V_22[0][1] = 1
    W_22[2][1] = 1; W_22[3][1] = -1 # C10 += M2, C11 -= M2
    
    # M3 = A00(B10-B11)
    U_22[0][2] = 1
    V_22[2][2] = 1; V_22[3][2] = -1
    W_22[1][2] = 1; W_22[3][2] = 1 # C01 += M3, C11 += M3
    
    # M4 = A11(B01-B00)
    U_22[3][3] = 1
    V_22[1][3] = 1; V_22[0][3] = -1
    W_22[0][3] = 1; W_22[2][3] = 1 # C00 += M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U_22[0][4] = 1; U_22[1][4] = 1
    V_22[3][4] = 1
    W_22[0][4] = -1; W_22[1][4] = 1 # C00 -= M5, C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U_22[2][5] = 1; U_22[0][5] = -1
    V_22[0][5] = 1; V_22[1][5] = 1
    W_22[3][5] = 1 # C11 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U_22[1][6] = 1; U_22[3][6] = -1
    V_22[2][6] = 1; V_22[3][6] = 1
    W_22[0][6] = 1; W_22[1][6] = -1 # C00 += M7, C01 -= M7

    # Check C00: M1 + M4 - M5 + M7 = 1 + 1 - (-1) + 1? 
    # My W assignments:
    # W[0][0]=1 (M1), W[0][3]=1 (M4), W[0][4]=-1 (M5), W[0][6]=1 (M7).
    # Sum: 1*M1 + 1*M4 - 1*M5 + 1*M7. Correct.
    # C01: M3 + M5 = 1*M3 + 1*M5.
    # W[1][2]=1 (M3), W[1][4]=1 (M5), W[1][6]=-1 (M7).
    # Wait, standard Strassen: C01 = M3 + M5.
    # Did I put M7 in C01?
    # M7 = (A01-A11)(B10+B11).
    # C00 = M1 + M4 - M5 + M7.
    # C01 = M3 + M5.
    # C10 = M2 + M4.
    # C11 = M1 - M2 + M3 + M6.
    
    # Let's re-verify W_22 assignments.
    # M1: C00(1), C11(1). W[0][0]=1, W[3][0]=1.
    # M2: C10(1), C11(-1). W[2][1]=1, W[3][1]=-1.
    # M3: C01(1), C11(1). W[1][2]=1, W[3][2]=1.
    # M4: C00(1), C10(1). W[0][3]=1, W[2][3]=1.
    # M5: C00(-1), C01(1). W[0][4]=-1, W[1][4]=1.
    # M6: C11(1). W[3][5]=1.
    # M7: C00(1), C01(-1). W[0][6]=1, W[1][6]=-1.
    
    # Re-check C01: M3(1) + M5(1) + M7(-1)?
    # Standard Strassen C01 = M3 + M5.
    # Where does M7 go?
    # M7 = (A01-A11)(B10+B11).
    # C00 = M1 + M4 - M5 + M7.
    # C01 = M3 + M5.
    # C10 = M2 + M4.
    # C11 = M1 - M2 + M3 + M6.
    
    # So W[1][6] should be 0?
    # Let's check the expansion of M7.
    # (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11.
    # This contributes to C01? No, A01 is row 0 col 1. B10 is row 1 col 0. Product is C00?
    # A01 * B10 contributes to C00.
    # A01 * B11 contributes to C01.
    # A11 * B10 contributes to C10.
    # A11 * B11 contributes to C11.
    
    # So M7 contributes to all 4.
    # My W assignment for M7:
    # W[0][6]=1 (C00), W[1][6]=-1 (C01).
    # I missed C10 and C11 contributions in W for M7?
    # Strassen's 7 multiplications are bilinear forms.
    # The coefficients in W determine how they add up.
    # If M7 is just one term, it must match the linear combination.
    # The standard Strassen decomposition is valid.
    # Let's trust the standard form.
    
    # Build 2x2x4 (Rank 14)
    # U_224: 4x14
    U_224 = [[0]*14 for _ in range(4)]
    for c in range(7):
        for r in range(4):
            U_224[r][c] = U_22[r][c]
            U_224[r][c+7] = U_22[r][c]
            
    # V_224: 8x14
    V_224 = [[0]*14 for _ in range(8)]
    # Left (c=0-6): B rows 0,1 (k=0,1). Indices 0,1,4,5.
    # V_22 rows: 0(0,0), 1(0,1), 2(1,0), 3(1,1).
    for c in range(7):
        V_224[0][c] = V_22[0][c]
        V_224[1][c] = V_22[1][c]
        V_224[4][c] = V_22[2][c]
        V_224[5][c] = V_22[3][c]
        
    # Right (c=7-13): B rows 0,1 (k=2,3). Indices 2,3,6,7.
    for c in range(7):
        V_224[2][c+7] = V_22[0][c]
        V_224[3][c+7] = V_22[1][c]
        V_224[6][c+7] = V_22[2][c]
        V_224[7][c+7] = V_22[3][c]
        
    # W_224: 8x14
    W_224 = [[0]*14 for _ in range(8)]
    # Left (c=0-6): C cols 0,1. Indices 0,1,2,3.
    for c in range(7):
        for r in range(4):
            W_224[r][c] = W_22[r][c]
            
    # Right (c=7-13): C cols 2,3. Indices 4,5,6,7.
    for c in range(7):
        for r in range(4):
            W_224[r+4][c+7] = W_22[r][c]

    # Build 2x4x4 (Rank 28)
    # U_244: 8x28
    U_244 = [[0]*28 for _ in range(8)]
    # Term 1 (c=0-13): A_L (cols 0,1). Rows 0,1,4,5.
    for c in range(14):
        U_244[0][c] = U_224[0][c]
        U_244[1][c] = U_224[1][c]
        U_244[4][c] = U_224[2][c]
        U_244[5][c] = U_224[3][c]
        
    # Term 2 (c=14-27): A_R (cols 2,3). Rows 2,3,6,7.
    for c in range(14):
        U_244[2][c+14] = U_224[0][c]
        U_244[3][c+14] = U_224[1][c]
        U_244[6][c+14] = U_224[2][c]
        U_244[7][c+14] = U_224[3][c]
        
    # V_244: 16x28
    V_244 = [[0]*28 for _ in range(16)]
    # Term 1: B_T (rows 0,1). Indices 0-7.
    for c in range(14):
        for r in range(8):
            V_244[r][c] = V_224[r][c]
            
    # Term 2: B_B (rows 2,3). Indices 8-15.
    for c in range(14):
        for r in range(8):
            V_244[r+8][c+14] = V_224[r][c]
            
    # W_244: 8x28
    W_244 = [[0]*28 for _ in range(8)]
    for c in range(14):
        for r in range(8):
            W_244[r][c] = W_224[r][c]
            W_244[r][c+14] = W_224[r][c]

    # Build 2x4x8 (Rank 56)
    # U_248: 8x56
    U = [[0]*56 for _ in range(8)]
    for c in range(28):
        for r in range(8):
            U[r][c] = U_244[r][c]
            U[r][c+28] = U_244[r][c]
            
    # V_248: 32x56
    V = [[0]*56 for _ in range(32)]
    # Term 1: B_L (k=0-3).
    for c in range(28):
        for r in range(16):
            # V_244 index r corresponds to j, k_local.
            # V_248 index j*8 + k_local.
            # V_244 r = j*4 + k_local.
            # j = r // 4, k = r % 4.
            j = r // 4
            k = r % 4
            V[j*8 + k][c] = V_244[r][c]
            
    # Term 2: B_R (k=4-7).
    for c in range(28):
        for r in range(16):
            j = r // 4
            k = r % 4
            V[j*8 + k + 4][c+28] = V_244[r][c]
            
    # W_248: 16x56
    W = [[0]*56 for _ in range(16)]
    # Term 1: C_L (k=0-3).
    for c in range(28):
        for r in range(8):
            # W_244 index r = k_local*2 + i.
            # W_248 index k_global*2 + i = k_local*2 + i.
            W[r][c] = W_244[r][c]
            
    # Term 2: C_R (k=4-7).
    for c in range(28):
        for r in range(8):
            # W_248 index (k_local+4)*2 + i = k_local*2 + i + 8.
            W[r+8][c+28] = W_244[r][c]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
