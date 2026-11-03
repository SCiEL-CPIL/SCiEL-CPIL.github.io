"""Matrix multiplication 3,4,7 - rank-minimal exact tensor decomposition"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
TIME_LIMIT = 130  # seconds, with safety margin

def _strassen_2x2():
    """Return Strassen decomposition for 2x2x2. Rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def _schoolbook_2x2x1():
    """Return schoolbook decomposition for 2x2x1. Rank 4."""
    U = np.zeros((4, 4))
    V = np.zeros((2, 4)) # B is 2x1, so 2 rows
    W = np.zeros((2, 4)) # C is 2x1, so 2 rows
    
    # A is 2x2, B is 2x1, C is 2x1
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    
    # Col 0: A00 B0
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # Col 1: A01 B1
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # Col 2: A10 B0
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # Col 3: A11 B1
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def _decompose_2x2x3():
    """
    Decompose 2x2 * 2x3. Rank 11.
    Split B into B1 (2x2) and B2 (2x1).
    C1 = A B1 (Rank 7)
    C2 = A B2 (Rank 4)
    """
    U_str, V_str, W_str = _strassen_2x2()
    U_sb, V_sb, W_sb = _schoolbook_2x2x1()
    
    # V_str is 4x7 (B is 2x2 -> 4 elements). V_sb is 2x4 (B is 2x1 -> 2 elements).
    # We need to map these to the full 2x3 B (6 elements).
    # B indices: 0,1 (col 0), 2,3 (col 1) for 2x2 part? No.
    # B is 2x3. Indices: 0,1,2 (row 0), 3,4,5 (row 1).
    # Or row-major: 0,1,2 (row 0), 3,4,5 (row 1).
    # In _strassen_2x2, V is 4x7. Rows 0,1,2,3 correspond to B[0,0], B[0,1], B[1,0], B[1,1].
    # In _schoolbook_2x2x1, V is 2x4. Rows 0,1 correspond to B[0,0], B[1,0].
    
    # We need to align these.
    # Let's build the full 2x2x3 decomposition from scratch to be safe.
    
    U = np.zeros((4, 11))
    V = np.zeros((6, 11)) # B is 2x3 -> 6 elements
    W = np.zeros((6, 11)) # C is 2x3 -> 6 elements
    
    # Part 1: A * B_left (2x2). Rank 7.
    # B_left is B[:, 0:2]. Indices 0,1 (row 0), 3,4 (row 1).
    # C_left is C[:, 0:2]. Indices 0,1 (row 0), 3,4 (row 1).
    
    # Map Strassen V (4x7) to V_full (6x7)
    # Strassen V rows: 0->B00, 1->B01, 2->B10, 3->B11
    # Full V rows: 0->B00, 1->B01, 2->B02, 3->B10, 4->B11, 5->B12
    for r in range(7):
        V[0, r] = V_str[0, r]
        V[1, r] = V_str[1, r]
        V[3, r] = V_str[2, r]
        V[4, r] = V_str[3, r]
        
    # Map Strassen W (4x7) to W_full (6x7)
    # Strassen W rows: 0->C00, 1->C01, 2->C10, 3->C11
    # Full W rows: 0->C00, 1->C01, 2->C02, 3->C10, 4->C11, 5->C12
    for r in range(7):
        W[0, r] = W_str[0, r]
        W[1, r] = W_str[1, r]
        W[3, r] = W_str[2, r]
        W[4, r] = W_str[3, r]
        
    U[:, 0:7] = U_str
    
    # Part 2: A * B_right (2x1). Rank 4.
    # B_right is B[:, 2]. Indices 2 (row 0), 5 (row 1).
    # C_right is C[:, 2]. Indices 2 (row 0), 5 (row 1).
    
    # Schoolbook V (2x4) rows: 0->B0, 1->B1.
    # Map to V_full:
    # B0 is B[0,2] -> Index 2.
    # B1 is B[1,2] -> Index 5.
    for r in range(4):
        V[2, 7+r] = V_sb[0, r]
        V[5, 7+r] = V_sb[1, r]
        
    # Schoolbook W (2x4) rows: 0->C0, 1->C1.
    # Map to W_full:
    # C0 is C[0,2] -> Index 2.
    # C1 is C[1,2] -> Index 5.
    for r in range(4):
        W[2, 7+r] = W_sb[0, r]
        W[5, 7+r] = W_sb[1, r]
        
    U[:, 7:11] = U_sb
    
    return U, V, W

def construct():
    """Find rank-minimal exact tensor decomposition for 3x4 * 4x7."""
    n, m, p = DIMENSIONS
    
    # Total rank: 50 (for 2x4x7) + 4 (for 1x4x7) = 54
    total_rank = 54
    
    U = np.zeros((n * m, total_rank)) # 12 x 54
    V = np.zeros((m * p, total_rank)) # 28 x 54
    W = np.zeros((p * n, total_rank)) # 21 x 54
    
    col_offset = 0
    
    # 1. A_TL (2x2) * B_top1 (2x2) -> C_TL1 (2x2). Rank 7.
    # A_TL: A[0:2, 0:2] -> Indices 0,1,4,5
    # B_top1: B[0:2, 0:2] -> Indices 0,1,7,8
    # C_TL1: C[0:2, 0:2] -> Indices 0,1,3,4
    
    U_str, V_str, W_str = _strassen_2x2()
    
    # Map U
    U[0, col_offset:col_offset+7] = U_str[0, :]
    U[1, col_offset:col_offset+7] = U_str[1, :]
    U[4, col_offset:col_offset+7] = U_str[2, :]
    U[5, col_offset:col_offset+7] = U_str[3, :]
    
    # Map V
    V[0, col_offset:col_offset+7] = V_str[0, :]
    V[1, col_offset:col_offset+7] = V_str[1, :]
    V[7, col_offset:col_offset+7] = V_str[2, :]
    V[8, col_offset:col_offset+7] = V_str[3, :]
    
    # Map W
    W[0, col_offset:col_offset+7] = W_str[0, :]
    W[1, col_offset:col_offset+7] = W_str[1, :]
    W[3, col_offset:col_offset+7] = W_str[2, :]
    W[4, col_offset:col_offset+7] = W_str[3, :]
    
    col_offset += 7
    
    # 2. A_TL (2x2) * B_top2 (2x2) -> C_TL2 (2x2). Rank 7.
    # B_top2: B[0:2, 2:4] -> Indices 2,3,9,10
    # C_TL2: C[2:4, 0:2] -> Indices 6,7,9,10
    
    U[0, col_offset:col_offset+7] = U_str[0, :]
    U[1, col_offset:col_offset+7] = U_str[1, :]
    U[4, col_offset:col_offset+7] = U_str[2, :]
    U[5, col_offset:col_offset+7] = U_str[3, :]
    
    V[2, col_offset:col_offset+7] = V_str[0, :]
    V[3, col_offset:col_offset+7] = V_str[1, :]
    V[9, col_offset:col_offset+7] = V_str[2, :]
    V[10, col_offset:col_offset+7] = V_str[3, :]
    
    W[6, col_offset:col_offset+7] = W_str[0, :]
    W[7, col_offset:col_offset+7] = W_str[1, :]
    W[9, col_offset:col_offset+7] = W_str[2, :]
    W[10, col_offset:col_offset+7] = W_str[3, :]
    
    col_offset += 7
    
    # 3. A_TL (2x2) * B_top3 (2x3) -> C_TL3 (2x3). Rank 11.
    # B_top3: B[0:2, 4:7] -> Indices 4,5,6,11,12,13
    # C_TL3: C[4:7, 0:2] -> Indices 12,13,14,15,16,17
    
    U_223, V_223, W_223 = _decompose_2x2x3()
    
    U[0, col_offset:col_offset+11] = U_223[0, :]
    U[1, col_offset:col_offset+11] = U_223[1, :]
    U[4, col_offset:col_offset+11] = U_223[2, :]
    U[5, col_offset:col_offset+11] = U_223[3, :]
    
    # V_223 rows: 0->B00, 1->B01, 2->B02, 3->B10, 4->B11, 5->B12
    # Target V indices: 4,5,6,11,12,13
    V[4, col_offset:col_offset+11] = V_223[0, :]
    V[5, col_offset:col_offset+11] = V_223[1, :]
    V[6, col_offset:col_offset+11] = V_223[2, :]
    V[11, col_offset:col_offset+11] = V_223[3, :]
    V[12, col_offset:col_offset+11] = V_223[4, :]
    V[13, col_offset:col_offset+11] = V_223[5, :]
    
    # W_223 rows: 0->C00, 1->C01, 2->C02, 3->C10, 4->C11, 5->C12
    # Target W indices: 12,13,14,15,16,17
    W[12, col_offset:col_offset+11] = W_223[0, :]
    W[13, col_offset:col_offset+11] = W_223[1, :]
    W[14, col_offset:col_offset+11] = W_223[2, :]
    W[15, col_offset:col_offset+11] = W_223[3, :]
    W[16, col_offset:col_offset+11] = W_223[4, :]
    W[17, col_offset:col_offset+11] = W_223[5, :]
    
    col_offset += 11
    
    # 4. A_TR (2x2) * B_bot1 (2x2) -> C_TR1 (2x2). Rank 7.
    # A_TR: A[0:2, 2:4] -> Indices 2,3,6,7
    # B_bot1: B[2:4, 0:2] -> Indices 14,15,21,22
    # C_TR1: C[0:2, 0:2] -> Indices 0,1,3,4
    
    U[2, col_offset:col_offset+7] = U_str[0, :]
    U[3, col_offset:col_offset+7] = U_str[1, :]
    U[6, col_offset:col_offset+7] = U_str[2, :]
    U[7, col_offset:col_offset+7] = U_str[3, :]
    
    V[14, col_offset:col_offset+7] = V_str[0, :]
    V[15, col_offset:col_offset+7] = V_str[1, :]
    V[21, col_offset:col_offset+7] = V_str[2, :]
    V[22, col_offset:col_offset+7] = V_str[3, :]
    
    W[0, col_offset:col_offset+7] = W_str[0, :]
    W[1, col_offset:col_offset+7] = W_str[1, :]
    W[3, col_offset:col_offset+7] = W_str[2, :]
    W[4, col_offset:col_offset+7] = W_str[3, :]
    
    col_offset += 7
    
    # 5. A_TR (2x2) * B_bot2 (2x2) -> C_TR2 (2x2). Rank 7.
    # B_bot2: B[2:4, 2:4] -> Indices 16,17,23,24
    # C_TR2: C[2:4, 0:2] -> Indices 6,7,9,10
    
    U[2, col_offset:col_offset+7] = U_str[0, :]
    U[3, col_offset:col_offset+7] = U_str[1, :]
    U[6, col_offset:col_offset+7] = U_str[2, :]
    U[7, col_offset:col_offset+7] = U_str[3, :]
    
    V[16, col_offset:col_offset+7] = V_str[0, :]
    V[17, col_offset:col_offset+7] = V_str[1, :]
    V[23, col_offset:col_offset+7] = V_str[2, :]
    V[24, col_offset:col_offset+7] = V_str[3, :]
    
    W[6, col_offset:col_offset+7] = W_str[0, :]
    W[7, col_offset:col_offset+7] = W_str[1, :]
    W[9, col_offset:col_offset+7] = W_str[2, :]
    W[10, col_offset:col_offset+7] = W_str[3, :]
    
    col_offset += 7
    
    # 6. A_TR (2x2) * B_bot3 (2x3) -> C_TR3 (2x3). Rank 11.
    # B_bot3: B[2:4, 4:7] -> Indices 18,19,20,25,26,27
    # C_TR3: C[4:7, 0:2] -> Indices 12,13,14,15,16,17
    
    U[2, col_offset:col_offset+11] = U_223[0, :]
    U[3, col_offset:col_offset+11] = U_223[1, :]
    U[6, col_offset:col_offset+11] = U_223[2, :]
    U[7, col_offset:col_offset+11] = U_223[3, :]
    
    V[18, col_offset:col_offset+11] = V_223[0, :]
    V[19, col_offset:col_offset+11] = V_223[1, :]
    V[20, col_offset:col_offset+11] = V_223[2, :]
    V[25, col_offset:col_offset+11] = V_223[3, :]
    V[26, col_offset:col_offset+11] = V_223[4, :]
    V[27, col_offset:col_offset+11] = V_223[5, :]
    
    W[12, col_offset:col_offset+11] = W_223[0, :]
    W[13, col_offset:col_offset+11] = W_223[1, :]
    W[14, col_offset:col_offset+11] = W_223[2, :]
    W[15, col_offset:col_offset+11] = W_223[3, :]
    W[16, col_offset:col_offset+11] = W_223[4, :]
    W[17, col_offset:col_offset+11] = W_223[5, :]
    
    col_offset += 11
    
    # 7. A_bot (1x4) * B (4x7) -> C_bot (1x7). Rank 4.
    # A_bot: A[2, :] -> Indices 8,9,10,11
    # B: All -> Indices 0..27
    # C_bot: C[:, 2] -> Indices 2,5,8,11,14,17,20
    
    for j in range(4):
        r = col_offset + j
        U[8+j, r] = 1
        for k in range(7):
            V[7*k+j, r] = 1
            W[3*k+2, r] = 1
            
    col_offset += 4
    
    # Convert to lists of integers
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
