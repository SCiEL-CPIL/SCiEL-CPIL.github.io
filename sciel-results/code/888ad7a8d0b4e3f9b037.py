import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    # 1. Define Strassen 2x2 U, V, W
    # Local A indices: 0,1,2,3
    # Local B indices: 0,1,2,3
    # Local C indices: 0,1,2,3
    # 7 Mults
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U: 4x7
    U_local = np.zeros((4, 7))
    V_local = np.zeros((4, 7))
    W_local = np.zeros((4, 7))
    
    # M1
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1; W_local[1, 0] = 0; W_local[2, 0] = 0; W_local[3, 0] = 1 # C00, C11
    
    # M2
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1; W_local[3, 1] = -1 # C10, C11 (Wait, C11 = M1 - M2 + M6 + M7)
    # Let's verify C formulas again.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    
    # Reset W to 0 and fill correctly
    W_local = np.zeros((4, 7))
    
    # M1
    W_local[0, 0] = 1
    W_local[3, 0] = 1
    
    # M2
    W_local[2, 1] = 1
    W_local[3, 1] = -1
    
    # M3
    W_local[1, 2] = 1
    
    # M4
    W_local[0, 3] = 1
    W_local[2, 3] = 1
    
    # M5
    W_local[0, 4] = -1
    W_local[1, 4] = 1
    
    # M6
    W_local[3, 5] = 1
    
    # M7
    W_local[0, 6] = 1
    W_local[3, 6] = 1
    
    # Verify U, V
    U_local = np.zeros((4, 7))
    V_local = np.zeros((4, 7))
    
    # M1: (A0+A3)(B0+B3)
    U_local[0,0]=1; U_local[3,0]=1
    V_local[0,0]=1; V_local[3,0]=1
    
    # M2: (A2+A3)B0
    U_local[2,1]=1; U_local[3,1]=1
    V_local[0,1]=1
    
    # M3: A0(B2-B3)
    U_local[0,2]=1
    V_local[2,2]=1; V_local[3,2]=-1
    
    # M4: A3(B1-B0)
    U_local[3,3]=1
    V_local[1,3]=1; V_local[0,3]=-1
    
    # M5: (A0+A1)B3
    U_local[0,4]=1; U_local[1,4]=1
    V_local[3,4]=1
    
    # M6: (A2-A0)(B0+B1)
    U_local[2,5]=1; U_local[0,5]=-1
    V_local[0,5]=1; V_local[1,5]=1
    
    # M7: (A1-A3)(B2+B3)
    U_local[1,6]=1; U_local[3,6]=-1
    V_local[2,6]=1; V_local[3,6]=1
    
    # Check: C00 = M1 + M4 - M5 + M7
    # W[0] should be [1, 0, 0, 1, -1, 0, 1]
    # My W_local[0] is [1, 0, 0, 1, -1, 0, 1]. Correct.
    # C01 = M3 + M5
    # W[1] should be [0, 0, 1, 0, 1, 0, 0]
    # My W_local[1] is [0, 0, 1, 0, 1, 0, 0]. Correct.
    # C10 = M2 + M4
    # W[2] should be [0, 1, 0, 1, 0, 0, 0]
    # My W_local[2] is [0, 1, 0, 1, 0, 0, 0]. Correct.
    # C11 = M1 - M2 + M6 + M7
    # W[3] should be [1, -1, 0, 0, 0, 1, 1]
    # My W_local[3] is [1, -1, 0, 0, 0, 1, 1]. Correct.

    # 2. Build 4x4 x 4x4 part (Rank 49)
    # Global A size 16, B size 16, C size 16
    # 7 steps, each 7 mults
    
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    # Step 1: L_A = A00 + A11, L_B = B00 + B11
    # A00: 0,1,4,5. A11: 10,11,14,15
    # B00: 0,1,4,5. B11: 10,11,14,15
    # C: C00 gets M1, C11 gets M1
    # C00: 0,1,4,5. C11: 10,11,14,15
    
    # Mapping A_global -> A_local (4)
    # A_local[0] = A_global[0] (A00[0])
    # A_local[1] = A_global[1] (A00[1])
    # A_local[2] = A_global[4] (A00[2])
    # A_local[3] = A_global[5] (A00[3])
    # Plus A11 contributions?
    # L_A = A00 + A11.
    # So L_A[0] = A00[0] + A11[0] = A_global[0] + A_global[10]
    # L_A[1] = A00[1] + A11[1] = A_global[1] + A_global[11]
    # L_A[2] = A00[2] + A11[2] = A_global[4] + A_global[14]
    # L_A[3] = A00[3] + A11[3] = A_global[5] + A_global[15]
    
    # We can form a selection matrix M_A (16x4) such that L_A = M_A @ A_global
    M_A1 = np.zeros((16, 4))
    M_A1[0, 0] = 1; M_A1[10, 0] = 1
    M_A1[1, 1] = 1; M_A1[11, 1] = 1
    M_A1[4, 2] = 1; M_A1[14, 2] = 1
    M_A1[5, 3] = 1; M_A1[15, 3] = 1
    
    M_B1 = np.zeros((16, 4))
    M_B1[0, 0] = 1; M_B1[10, 0] = 1
    M_B1[1, 1] = 1; M_B1[11, 1] = 1
    M_B1[4, 2] = 1; M_B1[14, 2] = 1
    M_B1[5, 3] = 1; M_B1[15, 3] = 1
    
    # C mapping: C_local (4) -> C_global (16)
    # C00 is top-left. C11 is bot-right.
    # In Strassen 2x2, C_local[0] is C00, C_local[1] is C01, C_local[2] is C10, C_local[3] is C11.
    # We want C_global to receive C_local[0] at C00 and C_local[3] at C11.
    # M_C1 (16x4)
    M_C1 = np.zeros((16, 4))
    M_C1[0, 0] = 1; M_C1[1, 0] = 1; M_C1[4, 0] = 1; M_C1[5, 0] = 1 # C00
    M_C1[10, 3] = 1; M_C1[11, 3] = 1; M_C1[14, 3] = 1; M_C1[15, 3] = 1 # C11
    
    # Add to global
    U_44[:, 0:7] = M_A1 @ U_local
    V_44[:, 0:7] = M_B1 @ V_local
    W_44[:, 0:7] = M_C1 @ W_local

    # Step 2: L_A = A10 + A11, L_B = B00
    # A10: 8,9,12,13. A11: 10,11,14,15
    M_A2 = np.zeros((16, 4))
    M_A2[8, 0] = 1; M_A2[10, 0] = 1
    M_A2[9, 1] = 1; M_A2[11, 1] = 1
    M_A2[12, 2] = 1; M_A2[14, 2] = 1
    M_A2[13, 3] = 1; M_A2[15, 3] = 1
    
    M_B2 = np.zeros((16, 4))
    M_B2[0, 0] = 1; M_B2[1, 1] = 1; M_B2[4, 2] = 1; M_B2[5, 3] = 1
    
    # C: C10 gets M2, C11 gets -M2
    M_C2 = np.zeros((16, 4))
    M_C2[8, 2] = 1; M_C2[9, 2] = 1; M_C2[12, 2] = 1; M_C2[13, 2] = 1 # C10
    M_C2[10, 3] = -1; M_C2[11, 3] = -1; M_C2[14, 3] = -1; M_C2[15, 3] = -1 # C11
    
    U_44[:, 7:14] = M_A2 @ U_local
    V_44[:, 7:14] = M_B2 @ V_local
    W_44[:, 7:14] = M_C2 @ W_local

    # Step 3: L_A = A00, L_B = B10 - B11
    M_A3 = np.zeros((16, 4))
    M_A3[0, 0] = 1; M_A3[1, 1] = 1; M_A3[4, 2] = 1; M_A3[5, 3] = 1
    
    M_B3 = np.zeros((16, 4))
    # B10: 8,9,12,13. B11: 10,11,14,15
    M_B3[8, 0] = 1; M_B3[10, 0] = -1
    M_B3[9, 1] = 1; M_B3[11, 1] = -1
    M_B3[12, 2] = 1; M_B3[14, 2] = -1
    M_B3[13, 3] = 1; M_B3[15, 3] = -1
    
    # C: C01 gets M3
    M_C3 = np.zeros((16, 4))
    M_C3[2, 1] = 1; M_C3[3, 1] = 1; M_C3[6, 1] = 1; M_C3[7, 1] = 1 # C01
    
    U_44[:, 14:21] = M_A3 @ U_local
    V_44[:, 14:21] = M_B3 @ V_local
    W_44[:, 14:21] = M_C3 @ W_local

    # Step 4: L_A = A11, L_B = B01 - B00
    M_A4 = np.zeros((16, 4))
    M_A4[10, 0] = 1; M_A4[11, 1] = 1; M_A4[14, 2] = 1; M_A4[15, 3] = 1
    
    M_B4 = np.zeros((16, 4))
    # B01: 2,3,6,7. B00: 0,1,4,5
    M_B4[2, 0] = 1; M_B4[0, 0] = -1
    M_B4[3, 1] = 1; M_B4[1, 1] = -1
    M_B4[6, 2] = 1; M_B4[4, 2] = -1
    M_B4[7, 3] = 1; M_B4[5, 3] = -1
    
    # C: C00 gets M4, C10 gets M4
    M_C4 = np.zeros((16, 4))
    M_C4[0, 0] = 1; M_C4[1, 0] = 1; M_C4[4, 0] = 1; M_C4[5, 0] = 1 # C00
    M_C4[8, 2] = 1; M_C4[9, 2] = 1; M_C4[12, 2] = 1; M_C4[13, 2] = 1 # C10
    
    U_44[:, 21:28] = M_A4 @ U_local
    V_44[:, 21:28] = M_B4 @ V_local
    W_44[:, 21:28] = M_C4 @ W_local

    # Step 5: L_A = A00 + A01, L_B = B11
    M_A5 = np.zeros((16, 4))
    # A00: 0,1,4,5. A01: 2,3,6,7
    M_A5[0, 0] = 1; M_A5[2, 0] = 1
    M_A5[1, 1] = 1; M_A5[3, 1] = 1
    M_A5[4, 2] = 1; M_A5[6, 2] = 1
    M_A5[5, 3] = 1; M_A5[7, 3] = 1
    
    M_B5 = np.zeros((16, 4))
    M_B5[10, 0] = 1; M_B5[11, 1] = 1; M_B5[14, 2] = 1; M_B5[15, 3] = 1
    
    # C: C00 gets -M5, C01 gets M5
    M_C5 = np.zeros((16, 4))
    M_C5[0, 0] = -1; M_C5[1, 0] = -1; M_C5[4, 0] = -1; M_C5[5, 0] = -1 # C00
    M_C5[2, 1] = 1; M_C5[3, 1] = 1; M_C5[6, 1] = 1; M_C5[7, 1] = 1 # C01
    
    U_44[:, 28:35] = M_A5 @ U_local
    V_44[:, 28:35] = M_B5 @ V_local
    W_44[:, 28:35] = M_C5 @ W_local

    # Step 6: L_A = A10 - A00, L_B = B00 + B01
    M_A6 = np.zeros((16, 4))
    # A10: 8,9,12,13. A00: 0,1,4,5
    M_A6[8, 0] = 1; M_A6[0, 0] = -1
    M_A6[9, 1] = 1; M_A6[1, 1] = -1
    M_A6[12, 2] = 1; M_A6[4, 2] = -1
    M_A6[13, 3] = 1; M_A6[5, 3] = -1
    
    M_B6 = np.zeros((16, 4))
    # B00: 0,1,4,5. B01: 2,3,6,7
    M_B6[0, 0] = 1; M_B6[2, 0] = 1
    M_B6[1, 1] = 1; M_B6[3, 1] = 1
    M_B6[4, 2] = 1; M_B6[6, 2] = 1
    M_B6[5, 3] = 1; M_B6[7, 3] = 1
    
    # C: C11 gets M6
    M_C6 = np.zeros((16, 4))
    M_C6[10, 3] = 1; M_C6[11, 3] = 1; M_C6[14, 3] = 1; M_C6[15, 3] = 1 # C11
    
    U_44[:, 35:42] = M_A6 @ U_local
    V_44[:, 35:42] = M_B6 @ V_local
    W_44[:, 35:42] = M_C6 @ W_local

    # Step 7: L_A = A01 - A11, L_B = B10 + B11
    M_A7 = np.zeros((16, 4))
    # A01: 2,3,6,7. A11: 10,11,14,15
    M_A7[2, 0] = 1; M_A7[10, 0] = -1
    M_A7[3, 1] = 1; M_A7[11, 1] = -1
    M_A7[6, 2] = 1; M_A7[14, 2] = -1
    M_A7[7, 3] = 1; M_A7[15, 3] = -1
    
    M_B7 = np.zeros((16, 4))
    # B10: 8,9,12,13. B11: 10,11,14,15
    M_B7[8, 0] = 1; M_B7[10, 0] = 1
    M_B7[9, 1] = 1; M_B7[11, 1] = 1
    M_B7[12, 2] = 1; M_B7[14, 2] = 1
    M_B7[13, 3] = 1; M_B7[15, 3] = 1
    
    # C: C00 gets M7, C11 gets M7
    M_C7 = np.zeros((16, 4))
    M_C7[0, 0] = 1; M_C7[1, 0] = 1; M_C7[4, 0] = 1; M_C7[5, 0] = 1 # C00
    M_C7[10, 3] = 1; M_C7[11, 3] = 1; M_C7[14, 3] = 1; M_C7[15, 3] = 1 # C11
    
    U_44[:, 42:49] = M_A7 @ U_local
    V_44[:, 42:49] = M_B7 @ V_local
    W_44[:, 42:49] = M_C7 @ W_local

    # 3. Build 4x4 x 4x2 part (Rank 28)
    # Global A 16, Global B 24 (4x6), Global C 24 (4x6)
    # We only touch B cols 4,5 and C cols 4,5.
    
    U_42 = np.zeros((16, 28))
    V_42 = np.zeros((24, 28))
    W_42 = np.zeros((24, 28))
    
    # Term 1: A1top * B1right
    # A1top: 0,1,4,5
    M_A_T1 = np.zeros((16, 4))
    M_A_T1[0, 0] = 1; M_A_T1[1, 1] = 1; M_A_T1[4, 2] = 1; M_A_T1[5, 3] = 1
    
    # B1right: B rows 0,1 cols 4,5 -> Indices 4,5,10,11
    M_B_T1 = np.zeros((24, 4))
    M_B_T1[4, 0] = 1; M_B_T1[5, 1] = 1; M_B_T1[10, 2] = 1; M_B_T1[11, 3] = 1
    
    # C1topright: C rows 0,1 cols 4,5 -> Indices 4,5,10,11
    M_C_T1 = np.zeros((24, 4))
    M_C_T1[4, 0] = 1; M_C_T1[5, 1] = 1; M_C_T1[10, 2] = 1; M_C_T1[11, 3] = 1
    
    U_42[:, 0:7] = M_A_T1 @ U_local
    V_42[:, 0:7] = M_B_T1 @ V_local
    W_42[:, 0:7] = M_C_T1 @ W_local

    # Term 2: A1bot * B1right
    # A1bot: 8,9,12,13
    M_A_T2 = np.zeros((16, 4))
    M_A_T2[8, 0] = 1; M_A_T2[9, 1] = 1; M_A_T2[12, 2] = 1; M_A_T2[13, 3] = 1
    
    # B1right: same
    # C1botright: C rows 2,3 cols 4,5 -> Indices 16,17,22,23
    M_C_T2 = np.zeros((24, 4))
    M_C_T2[16, 0] = 1; M_C_T2[17, 1] = 1; M_C_T2[22, 2] = 1; M_C_T2[23, 3] = 1
    
    U_42[:, 7:14] = M_A_T2 @ U_local
    V_42[:, 7:14] = M_B_T1 @ V_local
    W_42[:, 7:14] = M_C_T2 @ W_local

    # Term 3: A2top * B2right
    # A2top: 2,3,6,7
    M_A_T3 = np.zeros((16, 4))
    M_A_T3[2, 0] = 1; M_A_T3[3, 1] = 1; M_A_T3[6, 2] = 1; M_A_T3[7, 3] = 1
    
    # B2right: B rows 2,3 cols 4,5 -> Indices 16,17,22,23
    M_B_T3 = np.zeros((24, 4))
    M_B_T3[16, 0] = 1; M_B_T3[17, 1] = 1; M_B_T3[22, 2] = 1; M_B_T3[23, 3] = 1
    
    # C1topright: same as T1
    M_C_T3 = M_C_T1
    
    U_42[:, 14:21] = M_A_T3 @ U_local
    V_42[:, 14:21] = M_B_T3 @ V_local
    W_42[:, 14:21] = M_C_T3 @ W_local

    # Term 4: A2bot * B2right
    # A2bot: 10,11,14,15
    M_A_T4 = np.zeros((16, 4))
    M_A_T4[10, 0] = 1; M_A_T4[11, 1] = 1; M_A_T4[14, 2] = 1; M_A_T4[15, 3] = 1
    
    # B2right: same
    # C1botright: same as T2
    M_C_T4 = M_C_T2
    
    U_42[:, 21:28] = M_A_T4 @ U_local
    V_42[:, 21:28] = M_B_T3 @ V_local
    W_42[:, 21:28] = M_C_T4 @ W_local

    # Combine
    U = np.hstack([U_44, U_42])
    V = np.hstack([V_44, V_42])
    W = np.hstack([W_44, W_42])
    
    return {'U': U, 'V': V, 'W': W}
