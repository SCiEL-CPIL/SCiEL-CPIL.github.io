# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 3, 4, 5
    
    # Decomposition of M_1,4,5 (Rank 4)
    # U (4, 4), V (20, 4), W (5, 4)
    U_145 = np.zeros((4, 4))
    V_145 = np.zeros((20, 4))
    W_145 = np.zeros((5, 4))
    for j in range(4):
        U_145[j, j] = 1
        for k in range(5):
            V_145[j * 5 + k, j] = 1
            W_145[k, j] = 1
            
    # Decomposition of M_2,2,2 (Rank 7, Strassen)
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # Strassen formulas
    # M1: (a11+a22)(b11+b22) -> c11, c22
    # M2: (a21+a22)b11 -> c21
    # M3: a11(b12-b22) -> c12
    # M4: a22(b21-b11) -> c21, c22
    # M5: (a11+a12)b22 -> c11, c12
    # M6: (a21-a11)(b11+b12) -> c22
    # M7: (a12-a22)(b21+b22) -> c11, c22
    
    # Indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # U (A)
    U_222[0, 0] = 1; U_222[3, 0] = 1 # M1: a11, a22
    U_222[2, 1] = 1; U_222[3, 1] = 1 # M2: a21, a22
    U_222[0, 2] = 1 # M3: a11
    U_222[3, 3] = 1 # M4: a22
    U_222[0, 4] = 1; U_222[1, 4] = 1 # M5: a11, a12
    U_222[2, 5] = 1; U_222[0, 5] = -1 # M6: a21, -a11
    U_222[1, 6] = 1; U_222[3, 6] = -1 # M7: a12, -a22
    
    # V (B)
    V_222[0, 0] = 1; V_222[3, 0] = 1 # M1: b11, b22
    V_222[0, 1] = 1 # M2: b11
    V_222[1, 2] = 1; V_222[3, 2] = -1 # M3: b12, -b22
    V_222[2, 3] = 1; V_222[0, 3] = -1 # M4: b21, -b11
    V_222[3, 4] = 1 # M5: b22
    V_222[0, 5] = 1; V_222[1, 5] = 1 # M6: b11, b12
    V_222[2, 6] = 1; V_222[3, 6] = 1 # M7: b21, b22
    
    # W (C)
    W_222[0, 0] = 1 # M1 -> c11
    W_222[3, 0] = 1 # M1 -> c22
    W_222[2, 1] = 1 # M2 -> c21
    W_222[1, 2] = 1 # M3 -> c12
    W_222[2, 3] = 1 # M4 -> c21
    W_222[3, 3] = 1 # M4 -> c22
    W_222[0, 4] = 1 # M5 -> c11
    W_222[1, 4] = 1 # M5 -> c12
    W_222[3, 5] = 1 # M6 -> c22
    W_222[0, 6] = 1 # M7 -> c11
    W_222[3, 6] = 1 # M7 -> c22
    
    # Wait, W indices are k*n + i.
    # n=2.
    # c11 (i=0,k=0) -> 0
    # c12 (i=0,k=1) -> 1
    # c21 (i=1,k=0) -> 2
    # c22 (i=1,k=1) -> 3
    
    # Correct W:
    W_222 = np.zeros((4, 7))
    # M1: c11, c22
    W_222[0, 0] = 1; W_222[3, 0] = 1
    # M2: c21
    W_222[2, 1] = 1
    # M3: c12
    W_222[1, 2] = 1
    # M4: c21, c22
    W_222[2, 3] = 1; W_222[3, 3] = 1
    # M5: c11, c12
    W_222[0, 4] = 1; W_222[1, 4] = 1
    # M6: c22
    W_222[3, 5] = 1
    # M7: c11, c22
    W_222[0, 6] = 1; W_222[3, 6] = 1

    # Decomposition of M_2,2,3 (Rank 12, Schoolbook)
    U_223 = np.zeros((4, 12))
    V_223 = np.zeros((6, 12))
    W_223 = np.zeros((6, 12))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U_223[i*2+j, col] = 1
                V_223[j*3+k, col] = 1
                W_223[k*2+i, col] = 1
                col += 1

    # Combine
    # Total Rank = 4 + 2*(7+12) = 4 + 38 = 42
    Rank = 42
    U = np.zeros((12, Rank))
    V = np.zeros((20, Rank))
    W = np.zeros((15, Rank))
    
    # Part 1: M_1,4,5 (Rows 0-3 of A, Cols 0-3 of B, Rows 0-4 of C)
    # A_1 is 1x4. B is 4x5. C_1 is 1x5.
    # U_145 is 4x4. V_145 is 20x4. W_145 is 5x4.
    # Map to global:
    # U: i=0, j=0..3 -> 0..3
    # V: j=0..3, k=0..4 -> 0..19
    # W: k=0..4, i=0 -> 0..4
    
    U[0:4, 0:4] = U_145
    V[0:20, 0:4] = V_145
    W[0:5, 0:4] = W_145
    
    # Part 2: M_2,4,5 (Rows 1-2 of A, Cols 0-3 of B, Rows 0-4 of C)
    # A_2 is 2x4. B is 4x5. C_2 is 2x5.
    # Split B into B_1 (2x5) and B_2 (2x5).
    # M_2,2,5 = M_2,2,2 + M_2,2,3.
    
    # Sub-part 2a: M_2,2,2 (A_2, B_1, C_2a)
    # A_2 rows 1-2 (global 4-7). B_1 cols 0-1 (global 0-5? No, B is 4x5).
    # B_1 is first 2 rows of B? No, B is 4x5.
    # We split m=4 into 2+2.
    # B_1 is rows 0-1 of B (4x5 -> 2x5).
    # B_2 is rows 2-3 of B (2x5).
    
    # M_2,2,2 uses B_1 (2x5? No, B_1 is 2x5, but M_2,2,2 is 2x2x2).
    # We need to split p=5 into 2+3.
    # B_11 is B[0:2, 0:2] (2x2).
    # B_12 is B[0:2, 2:5] (2x3).
    
    # So M_2,2,2 is for A_2 (2x2? No A_2 is 2x4).
    # A_21 is A[1:3, 0:2] (2x2).
    # A_22 is A[1:3, 2:4] (2x2).
    
    # This is getting complicated to map.
    # I will use the schoolbook for the whole 2x4x5 part to be safe, 
    # but that gives rank 40. Total 44.
    # 44 < 60. Score 1.36.
    
    # Let's just use schoolbook for M_2,4,5.
    U_245 = np.zeros((8, 40))
    V_245 = np.zeros((20, 40))
    W_245 = np.zeros((10, 40))
    col = 0
    for i in range(2):
        for j in range(4):
            for k in range(5):
                U_245[i*4+j, col] = 1
                V_245[j*5+k, col] = 1
                W_245[k*2+i, col] = 1
                col += 1
                
    # Map M_2,4,5 to global
    # A_2 is rows 1-2 of A (global 4-11).
    # B is global 0-19.
    # C_2 is rows 1-2 of C (global 5-14).
    
    U[4:12, 4:44] = U_245
    V[0:20, 4:44] = V_245
    W[5:15, 4:44] = W_245
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
