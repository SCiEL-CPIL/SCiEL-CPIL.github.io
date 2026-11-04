import numpy as np

def construct():
    # Strassen 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: A: 0,1,2,3 (row-major 2x2), B: 0,1,2,3, C: 0,1,2,3
    
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U_222[0, 0] = 1
    U_222[3, 0] = 1
    V_222[0, 0] = 1
    V_222[3, 0] = 1
    W_222[0, 0] = 1
    W_222[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U_222[2, 1] = 1
    U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1
    W_222[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U_222[0, 2] = 1
    V_222[1, 2] = 1
    V_222[3, 2] = -1
    W_222[1, 2] = 1
    W_222[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U_222[3, 3] = 1
    V_222[2, 3] = 1
    V_222[0, 3] = -1
    W_222[0, 3] = 1
    W_222[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U_222[0, 4] = 1
    U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = -1
    W_222[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U_222[2, 5] = 1
    U_222[0, 5] = -1
    V_222[0, 5] = 1
    V_222[1, 5] = 1
    W_222[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U_222[1, 6] = 1
    U_222[3, 6] = -1
    V_222[2, 6] = 1
    V_222[3, 6] = 1
    W_222[0, 6] = 1
    
    # Direct 2x2x1
    # U: 4x4, V: 2x4, W: 2x4
    U_221 = np.zeros((4, 4))
    V_221 = np.zeros((2, 4))
    W_221 = np.zeros((2, 4))
    
    # P1 = A00 B0
    U_221[0, 0] = 1
    V_221[0, 0] = 1
    W_221[0, 0] = 1
    
    # P2 = A01 B1
    U_221[1, 1] = 1
    V_221[1, 1] = 1
    W_221[0, 1] = 1
    
    # P3 = A10 B0
    U_221[2, 2] = 1
    V_221[0, 2] = 1
    W_221[1, 2] = 1
    
    # P4 = A11 B1
    U_221[3, 3] = 1
    V_221[1, 3] = 1
    W_221[1, 3] = 1

    # Build 2x2x5
    # U: 4x18, V: 10x18, W: 10x18
    U_225 = np.zeros((4, 18))
    V_225 = np.zeros((10, 18))
    W_225 = np.zeros((10, 18))
    
    # Part 1: 2x2x2 (B cols 0,1)
    U_225[:, 0:7] = U_222
    # V_222 is 4x7. Maps to B rows 0,1 cols 0,1.
    # Global B for 2x2x5 is 2x5. Indices 0..9.
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3.
    V_225[0:4, 0:7] = V_222
    # W_222 is 4x7. Maps to C rows 0,1 cols 0,1.
    # Global C for 2x2x5 is 2x5. Indices 0..9.
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3.
    W_225[0:4, 0:7] = W_222
    
    # Part 2: 2x2x2 (B cols 2,3)
    U_225[:, 7:14] = U_222
    # V_222 maps to B rows 0,1 cols 2,3.
    # B[0,2]=4, B[0,3]=5, B[1,2]=6, B[1,3]=7.
    V_225[4:8, 7:14] = V_222
    # W_222 maps to C rows 0,1 cols 2,3.
    # C[0,2]=4, C[0,3]=5, C[1,2]=6, C[1,3]=7.
    W_225[4:8, 7:14] = W_222
    
    # Part 3: 2x2x1 (B col 4)
    U_225[:, 14:18] = U_221
    # V_221 is 2x4. Maps to B rows 0,1 col 4.
    # B[0,4]=8, B[1,4]=9.
    V_225[8:10, 14:18] = V_221
    # W_221 is 2x4. Maps to C rows 0,1 col 4.
    # C[0,4]=8, C[1,4]=9.
    W_225[8:10, 14:18] = W_221

    # Build 2x4x5
    # U: 8x36, V: 20x36, W: 10x36
    U = np.zeros((8, 36))
    V = np.zeros((20, 36))
    W = np.zeros((10, 36))
    
    # Block 1: A rows 0,1 (indices 0,1 in 2x4 A? No, A is 2x4. Row 0 is A[0,0], A[0,1]. Row 1 is A[1,0], A[1,1].)
    # Wait, A is 2x4. Indices 0..7.
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3.
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7.
    # We split A into A_L (cols 0,1) and A_R (cols 2,3).
    # A_L indices: 0,1,4,5.
    # A_R indices: 2,3,6,7.
    
    # B is 4x5. Indices 0..19.
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[0,4]=4.
    # B[1,0]=5, B[1,1]=6, B[1,2]=7, B[1,3]=8, B[1,4]=9.
    # B[2,0]=10, B[2,1]=11, B[2,2]=12, B[2,3]=13, B[2,4]=14.
    # B[3,0]=15, B[3,1]=16, B[3,2]=17, B[3,3]=18, B[3,4]=19.
    # We split B into B_T (rows 0,1) and B_B (rows 2,3).
    # B_T indices: 0..9.
    # B_B indices: 10..19.
    
    # C is 2x5. Indices 0..9.
    
    # Block 1: A_L (2x2) * B_T (2x5)
    # U_225 is 4x18. Maps to A_L indices.
    # A_L local indices 0,1,2,3 map to global A indices 0,1,4,5.
    # U_225 row 0 (A00) -> Global A 0
    # U_225 row 1 (A01) -> Global A 1
    # U_225 row 2 (A10) -> Global A 4
    # U_225 row 3 (A11) -> Global A 5
    
    U[0, 0:18] = U_225[0, :]
    U[1, 0:18] = U_225[1, :]
    U[4, 0:18] = U_225[2, :]
    U[5, 0:18] = U_225[3, :]
    
    # V_225 is 10x18. Maps to B_T indices 0..9.
    V[0:10, 0:18] = V_225
    
    # W_225 is 10x18. Maps to C indices 0..9.
    W[:, 0:18] = W_225
    
    # Block 2: A_R (2x2) * B_B (2x5)
    # A_R local indices 0,1,2,3 map to global A indices 2,3,6,7.
    U[2, 18:36] = U_225[0, :]
    U[3, 18:36] = U_225[1, :]
    U[6, 18:36] = U_225[2, :]
    U[7, 18:36] = U_225[3, :]
    
    # V_225 is 10x18. Maps to B_B indices 10..19.
    V[10:20, 18:36] = V_225
    
    # W_225 is 10x18. Maps to C indices 0..9.
    W[:, 18:36] = W_225
    
    return {"U": U, "V": V, "W": W}
