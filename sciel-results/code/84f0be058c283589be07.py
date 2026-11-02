import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We decompose into:
    # A_L (2x2) * B_T (2x5)
    # A_R (2x2) * B_B (2x5)
    # B_T = [B_T2 (2x2), B_T3 (2x3)]
    # B_B = [B_B2 (2x2), B_B3 (2x3)]
    
    # Total rank = rank(2x2x2) + rank(2x2x3) + rank(2x2x2) + rank(2x2x3)
    # rank(2x2x2) = 7 (Strassen)
    # rank(2x2x3) = 12 (Schoolbook)
    # Total = 7 + 12 + 7 + 12 = 38
    
    rank = 38
    U = [[0] * rank for _ in range(n * m)] # 8x38
    V = [[0] * rank for _ in range(m * p)] # 20x38
    W = [[0] * rank for _ in range(p * n)] # 10x38
    
    # Indices for A (2x4):
    # A_L: rows 0,1; cols 0,1 -> indices 0,1,2,3
    # A_R: rows 0,1; cols 2,3 -> indices 4,5,6,7
    # Note: U index is i*m + j.
    # A_L[0,0] is U[0], A_L[0,1] is U[1], A_L[1,0] is U[2], A_L[1,1] is U[3]
    # A_R[0,0] is U[4], A_R[0,1] is U[5], A_R[1,0] is U[6], A_R[1,1] is U[7]
    
    # Indices for B (4x5):
    # B_T: rows 0,1; cols 0..4 -> indices 0..4 (row 0), 5..9 (row 1)
    # B_B: rows 2,3; cols 0..4 -> indices 10..14 (row 2), 15..19 (row 3)
    # Note: V index is j*p + k.
    # B_T[0,0] is V[0], ..., B_T[0,4] is V[4]
    # B_T[1,0] is V[5], ..., B_T[1,4] is V[9]
    # B_B[0,0] is V[10], ..., B_B[0,4] is V[14]
    # B_B[1,0] is V[15], ..., B_B[1,4] is V[19]
    
    # Indices for W (5x2):
    # W index is k*n + i.
    # C is 2x5.
    # C[0,0] is W[0], C[0,1] is W[1], C[0,2] is W[2], C[0,3] is W[3], C[0,4] is W[4]
    # C[1,0] is W[5], C[1,1] is W[6], C[1,2] is W[7], C[1,3] is W[8], C[1,4] is W[9]
    
    col = 0
    
    # Helper to add a Strassen block for 2x2x2
    # A_idx: [a00, a01, a10, a11] indices in U
    # B_idx: [b00, b01, b10, b11] indices in V
    # W_idx: [w00, w01, w10, w11] indices in W
    def add_strassen(a_idx, b_idx, w_idx):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[3]][col] += 1
        W[w_idx[0]][col] += 1
        W[w_idx[3]][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a_idx[2]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[w_idx[1]][col] += 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[a_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        V[b_idx[3]][col] -= 1
        W[w_idx[2]][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[a_idx[3]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[0]][col] -= 1
        W[w_idx[1]][col] += 1
        W[w_idx[3]][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a_idx[0]][col] += 1
        U[a_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[w_idx[0]][col] -= 1
        W[w_idx[2]][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a_idx[2]][col] += 1
        U[a_idx[0]][col] -= 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        W[w_idx[3]][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a_idx[1]][col] += 1
        U[a_idx[3]][col] -= 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] += 1
        W[w_idx[0]][col] += 1
        W[w_idx[3]][col] += 1
        col += 1
        
    # Helper to add a Schoolbook block for 2x2x3
    # A_idx: [a00, a01, a10, a11]
    # B_idx: [b00, b01, b02, b10, b11, b12]
    # W_idx: [w00, w01, w02, w10, w11, w12]
    def add_schoolbook_223(a_idx, b_idx, w_idx):
        nonlocal col
        # C00 = A00B00 + A01B10
        U[a_idx[0]][col] = 1
        V[b_idx[0]][col] = 1
        W[w_idx[0]][col] = 1
        col += 1
        
        U[a_idx[1]][col] = 1
        V[b_idx[3]][col] = 1
        W[w_idx[0]][col] = 1
        col += 1
        
        # C01 = A00B01 + A01B11
        U[a_idx[0]][col] = 1
        V[b_idx[1]][col] = 1
        W[w_idx[1]][col] = 1
        col += 1
        
        U[a_idx[1]][col] = 1
        V[b_idx[4]][col] = 1
        W[w_idx[1]][col] = 1
        col += 1
        
        # C02 = A00B02 + A01B12
        U[a_idx[0]][col] = 1
        V[b_idx[2]][col] = 1
        W[w_idx[2]][col] = 1
        col += 1
        
        U[a_idx[1]][col] = 1
        V[b_idx[5]][col] = 1
        W[w_idx[2]][col] = 1
        col += 1
        
        # C10 = A10B00 + A11B10
        U[a_idx[2]][col] = 1
        V[b_idx[0]][col] = 1
        W[w_idx[3]][col] = 1
        col += 1
        
        U[a_idx[3]][col] = 1
        V[b_idx[3]][col] = 1
        W[w_idx[3]][col] = 1
        col += 1
        
        # C11 = A10B01 + A11B11
        U[a_idx[2]][col] = 1
        V[b_idx[1]][col] = 1
        W[w_idx[4]][col] = 1
        col += 1
        
        U[a_idx[3]][col] = 1
        V[b_idx[4]][col] = 1
        W[w_idx[4]][col] = 1
        col += 1
        
        # C12 = A10B02 + A11B12
        U[a_idx[2]][col] = 1
        V[b_idx[2]][col] = 1
        W[w_idx[5]][col] = 1
        col += 1
        
        U[a_idx[3]][col] = 1
        V[b_idx[5]][col] = 1
        W[w_idx[5]][col] = 1
        col += 1

    # Block 1: A_L * B_T2 (2x2x2)
    # A_L indices: 0,1,2,3
    # B_T2 indices: 0,1,5,6 (B[0,0], B[0,1], B[1,0], B[1,1])
    # W indices for C[:, 0:2]: 0,1,5,6 (C[0,0], C[0,1], C[1,0], C[1,1])
    add_strassen([0, 1, 2, 3], [0, 1, 5, 6], [0, 1, 5, 6])
    
    # Block 2: A_L * B_T3 (2x2x3)
    # A_L indices: 0,1,2,3
    # B_T3 indices: 2,3,4, 7,8,9 (B[0,2], B[0,3], B[0,4], B[1,2], B[1,3], B[1,4])
    # W indices for C[:, 2:5]: 2,3,4, 7,8,9 (C[0,2], C[0,3], C[0,4], C[1,2], C[1,3], C[1,4])
    add_schoolbook_223([0, 1, 2, 3], [2, 3, 4, 7, 8, 9], [2, 3, 4, 7, 8, 9])
    
    # Block 3: A_R * B_B2 (2x2x2)
    # A_R indices: 4,5,6,7
    # B_B2 indices: 10,11,15,16 (B[2,0], B[2,1], B[3,0], B[3,1])
    # W indices for C[:, 0:2]: 0,1,5,6
    add_strassen([4, 5, 6, 7], [10, 11, 15, 16], [0, 1, 5, 6])
    
    # Block 4: A_R * B_B3 (2x2x3)
    # A_R indices: 4,5,6,7
    # B_B3 indices: 12,13,14, 17,18,19 (B[2,2], B[2,3], B[2,4], B[3,2], B[3,3], B[3,4])
    # W indices for C[:, 2:5]: 2,3,4, 7,8,9
    add_schoolbook_223([4, 5, 6, 7], [12, 13, 14, 17, 18, 19], [2, 3, 4, 7, 8, 9])
    
    return {"U": U, "V": V, "W": W}
