import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 36
    rank = 36
    
    # Initialize U, V, W
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Define Strassen factors for 2x2x2
    # U_strassen: 4x7
    # V_strassen: 4x7
    # W_strassen: 4x7 (order: P00, P10, P01, P11)
    
    # We will build the 18-rank block for 2x2x5
    # Then place it twice.
    
    # 18-rank block factors
    # U_block: 4x18
    # V_block: 10x18
    # W_block: 10x18
    
    U_block = np.zeros((4, 18), dtype=int)
    V_block = np.zeros((10, 18), dtype=int)
    W_block = np.zeros((10, 18), dtype=int)
    
    # Strassen 1 (cols 0-6)
    # U
    U_block[0, 0] = 1; U_block[3, 0] = 1
    U_block[2, 1] = 1; U_block[3, 1] = 1
    U_block[0, 2] = 1
    U_block[3, 3] = 1
    U_block[0, 4] = 1; U_block[1, 4] = 1
    U_block[0, 5] = -1; U_block[2, 5] = 1
    U_block[1, 6] = 1; U_block[3, 6] = -1
    
    # V (rows: V00, V01, V02, V03, V04, V10, V11, V12, V13, V14)
    # Strassen uses V00, V01, V10, V11
    # V00 is row 0, V01 is row 1, V10 is row 5, V11 is row 6
    
    V_block[0, 0] = 1; V_block[6, 0] = 1
    V_block[0, 1] = 1
    V_block[1, 2] = 1; V_block[6, 2] = -1
    V_block[5, 3] = 1; V_block[0, 3] = -1
    V_block[6, 4] = 1
    V_block[0, 5] = 1; V_block[1, 5] = 1
    V_block[5, 6] = 1; V_block[6, 6] = 1
    
    # W (rows: P00, P10, P01, P11, P02, P12, P03, P13, P04, P14)
    # Strassen 1 affects P00, P10, P01, P11 (rows 0-3)
    
    W_block[0, 0] = 1; W_block[3, 0] = 1
    W_block[1, 1] = 1; W_block[3, 1] = -1
    W_block[2, 2] = 1; W_block[3, 2] = 1
    W_block[0, 3] = 1; W_block[1, 3] = 1
    W_block[0, 4] = -1; W_block[2, 4] = 1
    W_block[3, 5] = 1
    W_block[0, 6] = 1; W_block[3, 6] = 1
    
    # Strassen 2 (cols 7-13)
    # Same U
    U_block[:, 7:14] = U_block[:, 0:7]
    
    # V uses V02, V03, V12, V13
    # V02 is row 2, V03 is row 3, V12 is row 7, V13 is row 8
    # Map: 0->2, 1->3, 5->7, 6->8
    
    V_block[2, 7] = 1; V_block[8, 7] = 1
    V_block[2, 8] = 1
    V_block[3, 9] = 1; V_block[8, 9] = -1
    V_block[7, 10] = 1; V_block[2, 10] = -1
    V_block[8, 11] = 1
    V_block[2, 12] = 1; V_block[3, 12] = 1
    V_block[7, 13] = 1; V_block[8, 13] = 1
    
    # W affects P02, P12, P03, P13 (rows 4-7)
    # Map: 0->4, 1->5, 2->6, 3->7
    
    W_block[4, 7] = 1; W_block[7, 7] = 1
    W_block[5, 8] = 1; W_block[7, 8] = -1
    W_block[6, 9] = 1; W_block[7, 9] = 1
    W_block[4, 10] = 1; W_block[5, 10] = 1
    W_block[4, 11] = -1; W_block[6, 11] = 1
    W_block[7, 12] = 1
    W_block[4, 13] = 1; W_block[7, 13] = 1
    
    # Schoolbook (cols 14-17)
    # M00 V04, M01 V14, M10 V04, M11 V14
    
    U_block[0, 14] = 1
    U_block[1, 15] = 1
    U_block[2, 16] = 1
    U_block[3, 17] = 1
    
    # V uses V04 (row 4), V14 (row 9)
    V_block[4, 14] = 1
    V_block[9, 15] = 1
    V_block[4, 16] = 1
    V_block[9, 17] = 1
    
    # W affects P04 (row 8), P14 (row 9)
    # P04 = M00 V04 + M01 V14
    # P14 = M10 V04 + M11 V14
    
    W_block[8, 14] = 1
    W_block[8, 15] = 1
    W_block[9, 16] = 1
    W_block[9, 17] = 1
    
    # Now place Block 1 (LT) into U, V, W
    # U rows: 0, 1, 4, 5
    # V rows: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    # W rows: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    
    u_rows_1 = [0, 1, 4, 5]
    v_rows_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    w_rows_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for i, ur in enumerate(u_rows_1):
        U[ur, 0:18] = U_block[i, :]
    for i, vr in enumerate(v_rows_1):
        V[vr, 0:18] = V_block[i, :]
    for i, wr in enumerate(w_rows_1):
        W[wr, 0:18] = W_block[i, :]
        
    # Place Block 2 (RB) into U, V, W
    # U rows: 2, 3, 6, 7
    # V rows: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    # W rows: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (Same C)
    
    u_rows_2 = [2, 3, 6, 7]
    v_rows_2 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    w_rows_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for i, ur in enumerate(u_rows_2):
        U[ur, 18:36] = U_block[i, :]
    for i, vr in enumerate(v_rows_2):
        V[vr, 18:36] = V_block[i, :]
    for i, wr in enumerate(w_rows_2):
        W[wr, 18:36] = W_block[i, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
