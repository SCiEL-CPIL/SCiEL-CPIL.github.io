import numpy as np
import time

def construct():
    # Dimensions: A is 2x2, B is 2x4, C is 2x4
    n, m, p = 2, 2, 4
    R = 14
    
    # Initialize matrices
    U = np.zeros((n*m, R)) # 4x14
    V = np.zeros((m*p, R)) # 8x14
    W = np.zeros((p*n, R)) # 8x14
    
    # Strassen decomposition for 2x2 * 2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # Indices for A: 0:a00, 1:a01, 2:a10, 3:a11
    # Indices for B (block 1): 0:b00, 1:b01, 2:b10, 3:b11
    # Indices for B (block 2): 4:b02, 5:b03, 6:b12, 7:b13
    # Indices for C (block 1): 0:c00, 1:c01, 2:c10, 3:c11
    # Indices for C (block 2): 4:c02, 5:c03, 6:c12, 7:c13
    
    # Block 1: cols 0,1 of B
    # r=0: M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # c00 += M1, c11 += M1? 
    # Wait, standard Strassen:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's map the standard Strassen terms to the tensor decomposition.
    # Term r contributes u_r * v_r * w_r to C.
    # M1 = (A00+A11)(B00+B11)
    # A part: A00+A11 -> U[0]=1, U[3]=1
    # B part: B00+B11 -> V[0]=1, V[3]=1 (indices 0,1,2,3 for B block 1)
    # C part: contributes to C00 and C11.
    # C00 = M1 - M4 + M5 - M7
    # C11 = M1 - M2 + M3 + M6
    # So M1 adds 1 to C00 and 1 to C11.
    # W[0, 0] = 1 (C00), W[3, 0] = 1 (C11)
    
    # M2 = (A10+A11)B00
    # A part: A10+A11 -> U[2]=1, U[3]=1
    # B part: B00 -> V[0]=1
    # C part: contributes to C10 and C11.
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M2 adds 1 to C10 and -1 to C11.
    # W[2, 1] = 1 (C10), W[3, 1] = -1 (C11)
    # U[2, 1] = 1; U[3, 1] = 1
    # V[0, 1] = 1
    
    # M3 = A00(B01-B11)
    # A part: A00 -> U[0]=1
    # B part: B01-B11 -> V[1]=1, V[3]=-1
    # C part: contributes to C01 and C11.
    # C01 = M3 + M7
    # C11 = M1 - M2 + M3 + M6
    # So M3 adds 1 to C01 and 1 to C11.
    # W[1, 2] = 1 (C01), W[3, 2] = 1 (C11)
    # U[0, 2] = 1
    # V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11(B10-B00)
    # A part: A11 -> U[3]=1
    # B part: B10-B00 -> V[2]=1, V[0]=-1
    # C part: contributes to C00 and C10.
    # C00 = M1 - M4 + M5 - M7
    # C10 = M2 + M4
    # So M4 adds -1 to C00 and 1 to C10.
    # W[0, 3] = -1 (C00), W[2, 3] = 1 (C10)
    # U[3, 3] = 1
    # V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00+A01)B11
    # A part: A00+A01 -> U[0]=1, U[1]=1
    # B part: B11 -> V[3]=1
    # C part: contributes to C00 and C01.
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # Wait, C01 does not have M5.
    # Let's re-verify Strassen formulas.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M5 contributes to C00 only?
    # C00 has M5. C01, C10, C11 do not.
    # So M5 adds 1 to C00.
    # W[0, 4] = 1 (C00)
    # U[0, 4] = 1; U[1, 4] = 1
    # V[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # A part: A10-A00 -> U[2]=1, U[0]=-1
    # B part: B00+B01 -> V[0]=1, V[1]=1
    # C part: contributes to C11.
    # C11 has M6.
    # So M6 adds 1 to C11.
    # W[3, 5] = 1 (C11)
    # U[2, 5] = 1; U[0, 5] = -1
    # V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # A part: A01-A11 -> U[1]=1, U[3]=-1
    # B part: B10+B11 -> V[2]=1, V[3]=1
    # C part: contributes to C00 and C01.
    # C00 has -M7.
    # C01 has M7.
    # So M7 adds -1 to C00 and 1 to C01.
    # W[0, 6] = -1 (C00), W[1, 6] = 1 (C01)
    # U[1, 6] = 1; U[3, 6] = -1
    # V[2, 6] = 1; V[3, 6] = 1

    # Now fill the matrices for Block 1 (cols 0-6)
    
    # r=0 (M1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1 (M2)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # r=2 (M3)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # r=3 (M4)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # r=4 (M5)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1
    
    # r=5 (M6)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6 (M7)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1

    # Block 2: cols 2,3 of B
    # B indices: 4:b02, 5:b03, 6:b12, 7:b13
    # C indices: 4:c02, 5:c03, 6:c12, 7:c13
    
    # The A part is the same.
    # The B part is shifted by 4.
    # The C part is shifted by 4.
    
    # r=7 (M1 for block 2)
    U[0, 7] = 1; U[3, 7] = 1
    V[4, 7] = 1; V[7, 7] = 1
    W[4, 7] = 1; W[7, 7] = 1
    
    # r=8 (M2 for block 2)
    U[2, 8] = 1; U[3, 8] = 1
    V[4, 8] = 1
    W[6, 8] = 1; W[7, 8] = -1
    
    # r=9 (M3 for block 2)
    U[0, 9] = 1
    V[5, 9] = 1; V[7, 9] = -1
    W[5, 9] = 1; W[7, 9] = 1
    
    # r=10 (M4 for block 2)
    U[3, 10] = 1
    V[6, 10] = 1; V[4, 10] = -1
    W[4, 10] = -1; W[6, 10] = 1
    
    # r=11 (M5 for block 2)
    U[0, 11] = 1; U[1, 11] = 1
    V[7, 11] = 1
    W[4, 11] = 1
    
    # r=12 (M6 for block 2)
    U[2, 12] = 1; U[0, 12] = -1
    V[4, 12] = 1; V[5, 12] = 1
    W[7, 12] = 1
    
    # r=13 (M7 for block 2)
    U[1, 13] = 1; U[3, 13] = -1
    V[6, 13] = 1; V[7, 13] = 1
    W[4, 13] = -1; W[5, 13] = 1

    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
