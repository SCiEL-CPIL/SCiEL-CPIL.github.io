import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A
    # Indices: 0:A00, 1:A01, 2:A10, 3:A11
    S_forms = [
        [1, 0, 0, 1], # S1: A00 + A11
        [0, 0, 1, 1], # S2: A10 + A11
        [1, 0, 0, 0], # S3: A00
        [0, 0, 0, 1], # S4: A11
        [1, 1, 0, 0], # S5: A00 + A01
        [-1, 0, 1, 0],# S6: A10 - A00
        [0, 1, 0, -1] # S7: A01 - A11
    ]
    
    # Strassen reconstruction coefficients for C
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4 + M7
    # C10 = M2 + M4 - M5 + M6
    # C11 = M1 - M2 + M3 + M6
    # Indices: 0:C00, 1:C01, 2:C10, 3:C11
    W_forms = [
        [1, 0, 0, 0], # Coeffs for M1
        [0, 0, 1, 0], # Coeffs for M2
        [0, 1, 0, 0], # Coeffs for M3
        [-1, 1, 1, 0],# Coeffs for M4
        [1, 0, -1, 0],# Coeffs for M5
        [0, 0, 1, 1], # Coeffs for M6
        [-1, 1, 0, 1] # Coeffs for M7
    ]
    
    col_idx = 0
    # Process 7 pairs of columns
    for k in range(7):
        col_k = 2 * k
        col_k1 = 2 * k + 1
        
        # T forms for B pair k
        # B indices: 0:B0, 1:B1 (for row 0), 2:B2, 3:B3 (for row 1)
        # Relative to global B:
        # B[0, col_k], B[0, col_k1], B[1, col_k], B[1, col_k1]
        
        # T1: B00 + B11 -> B[0, col_k] + B[1, col_k1]
        # T2: B00 -> B[0, col_k]
        # T3: B10 - B11 -> B[1, col_k] - B[1, col_k1]
        # T4: B01 - B00 -> B[0, col_k1] - B[0, col_k]
        # T5: B11 -> B[1, col_k1]
        # T6: B00 + B10 -> B[0, col_k] + B[1, col_k]
        # T7: B11 + B01 -> B[1, col_k1] + B[0, col_k1]
        
        T_forms = [
            # [B[0,ck], B[0,ck1], B[1,ck], B[1,ck1]]
            [1, 0, 0, 1], # T1
            [1, 0, 0, 0], # T2
            [0, 0, 1, -1],# T3
            [-1, 1, 0, 0],# T4
            [0, 0, 0, 1], # T5
            [1, 0, 1, 0], # T6
            [0, 1, 0, 1]  # T7
        ]
        
        for j in range(7):
            c = col_idx
            
            # Fill U
            for i in range(4):
                U[i][c] = S_forms[j][i]
                
            # Fill V
            # V index is j_B * p + k_B
            # We need to set V for the 4 relevant B entries
            # T_forms[j] gives coeffs for [B[0,ck], B[0,ck1], B[1,ck], B[1,ck1]]
            
            # B[0, ck] -> index 0*p + ck = ck
            V[0 * p + col_k][c] = T_forms[j][0]
            # B[0, ck1] -> index 0*p + ck1 = ck1
            V[0 * p + col_k1][c] = T_forms[j][1]
            # B[1, ck] -> index 1*p + ck = p + ck
            V[1 * p + col_k][c] = T_forms[j][2]
            # B[1, ck1] -> index 1*p + ck1 = p + ck1
            V[1 * p + col_k1][c] = T_forms[j][3]
            
            # Fill W
            # W index is k_C * n + i_C
            # W_forms[j] gives coeffs for [C[0,ck], C[0,ck1], C[1,ck], C[1,ck1]]
            
            # C[0, ck] -> index ck*n + 0
            W[col_k * n + 0][c] = W_forms[j][0]
            # C[0, ck1] -> index ck1*n + 0
            W[col_k1 * n + 0][c] = W_forms[j][1]
            # C[1, ck] -> index ck*n + 1
            W[col_k * n + 1][c] = W_forms[j][2]
            # C[1, ck1] -> index ck1*n + 1
            W[col_k1 * n + 1][c] = W_forms[j][3]
            
            col_idx += 1
            
    # Process last column (k=14)
    # Schoolbook: 4 multiplications
    # P1 = A00 * B0,14
    # P2 = A01 * B1,14
    # P3 = A10 * B0,14
    # P4 = A11 * B1,14
    
    # C0,14 = P1 + P2
    # C1,14 = P3 + P4
    
    # Col 49: A00 * B0,14
    c = 49
    U[0][c] = 1 # A00
    V[0 * p + 14][c] = 1 # B0,14
    W[14 * n + 0][c] = 1 # C0,14
    W[14 * n + 1][c] = 0
    
    # Col 50: A01 * B1,14
    c = 50
    U[1][c] = 1 # A01
    V[1 * p + 14][c] = 1 # B1,14
    W[14 * n + 0][c] = 1 # C0,14
    W[14 * n + 1][c] = 0
    
    # Col 51: A10 * B0,14
    c = 51
    U[2][c] = 1 # A10
    V[0 * p + 14][c] = 1 # B0,14
    W[14 * n + 0][c] = 0
    W[14 * n + 1][c] = 1 # C1,14
    
    # Col 52: A11 * B1,14
    c = 52
    U[3][c] = 1 # A11
    V[1 * p + 14][c] = 1 # B1,14
    W[14 * n + 0][c] = 0
    W[14 * n + 1][c] = 1 # C1,14
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
