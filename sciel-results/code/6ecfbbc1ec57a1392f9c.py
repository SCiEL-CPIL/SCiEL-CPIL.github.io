import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Step 1: Strassen 2x2 decomposition (rank 7)
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = -1
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = -1
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1; W2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1; W2[3, 6] = -1
    
    # Step 2: Laderman 3x3 decomposition (rank 7)
    # U3[i*3+j, r], V3[j*3+k, r], W3[k*3+i, r]
    U3 = np.zeros((9, 7), dtype=float)
    V3 = np.zeros((9, 7), dtype=float)
    W3 = np.zeros((9, 7), dtype=float)
    
    # M1 = (a11 + a22)(b11 + b22)
    U3[0, 0] = 1; U3[4, 0] = 1  # a11=idx0, a22=idx4
    V3[0, 0] = 1; V3[4, 0] = 1  # b11=idx0, b22=idx4
    W3[0, 0] = 1; W3[4, 0] = 1  # c11=idx0, c22=idx4
    
    # M2 = (a12 + a23)(b12 + b21)
    U3[1, 1] = 1; U3[5, 1] = 1  # a12=idx1, a23=idx5
    V3[1, 1] = 1; V3[3, 1] = 1  # b12=idx1, b21=idx3
    W3[1, 1] = 1; W3[3, 1] = 1  # c12=idx1, c21=idx3
    
    # M3 = (a13 + a21)(b13 + b22)
    U3[2, 2] = 1; U3[3, 2] = 1  # a13=idx2, a21=idx3
    V3[2, 2] = 1; V3[4, 2] = 1  # b13=idx2, b22=idx4
    W3[2, 2] = 1; W3[4, 2] = 1  # c13=idx2, c23=idx5... wait
    
    # Let me be more careful with indices
    # For 3x3: A[i,j] has index i*3+j, B[j,k] has index j*3+k, C[k,i] has index k*3+i
    
    # Reset and redo carefully
    U3 = np.zeros((9, 7), dtype=float)
    V3 = np.zeros((9, 7), dtype=float)
    W3 = np.zeros((9, 7), dtype=float)
    
    # Laderman's 7 products for 3x3:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a01 + a12)(b01 + b10)
    # M3 = (a02 + a10)(b02 + b11)
    # M4 = (a11 - a22)(b11 + b22)
    # M5 = (a00 - a22)(b00 + b22)
    # M6 = (a02 - a21)(b20 + b22)
    # M7 = (a10 - a22)(b00 + b10)
    
    # Outputs:
    # c00 = M1 + M5 - M7
    # c01 = M2
    # c02 = M3 + M6
    # c10 = M7
    # c11 = M1 - M4 - M5
    # c12 = M4
    # c20 = M5
    # c21 = M6
    # c22 = M4 - M1 + M2 - M3
    
    # M1
    U3[0, 0] = 1; U3[4, 0] = 1  # a00=0, a11=4
    V3[0, 0] = 1; V3[4, 0] = 1  # b00=0, b11=4
    W3[0, 0] = 1; W3[4, 0] = 1  # c00=0, c11=4
    
    # M2
    U3[1, 1] = 1; U3[5, 1] = 1  # a01=1, a12=5
    V3[1, 1] = 1; V3[3, 1] = 1  # b01=1, b10=3
    W3[1, 1] = 1; W3[3, 1] = 1  # c01=1, c10=3
    
    # M3
    U3[2, 2] = 1; U3[3, 2] = 1  # a02=2, a10=3
    V3[2, 2] = 1; V3[4, 2] = 1  # b02=2, b11=4
    W3[2, 2] = 1; W3[5, 2] = 1  # c02=2, c12=5
    
    # M4
    U3[4, 3] = 1; U3[8, 3] = -1  # a11=4, a22=8
    V3[4, 3] = 1; V3[8, 3] = 1  # b11=4, b22=8
    W3[4, 3] = -1; W3[5, 3] = 1; W3[8, 3] = 1  # c11=4(-1), c12=5(1), c22=8(1)
    
    # M5
    U3[0, 4] = 1; U3[8, 4] = -1  # a00=0, a22=8
    V3[0, 4] = 1; V3[8, 4] = 1  # b00=0, b22=8
    W3[0, 4] = 1; W3[4, 4] = -1; W3[6, 4] = 1  # c00=0(1), c11=4(-1), c20=6(1)
    
    # M6
    U3[2, 5] = 1; U3[7, 5] = -1  # a02=2, a21=7
    V3[6, 5] = 1; V3[8, 5] = 1  # b20=6, b22=8
    W3[2, 5] = 1; W3[7, 5] = 1  # c02=2(1), c21=7(1)
    
    # M7
    U3[3, 6] = 1; U3[8, 6] = -1  # a10=3, a22=8
    V3[0, 6] = 1; V3[3, 6] = 1  # b00=0, b10=3
    W3[0, 6] = -1; W3[3, 6] = 1  # c00=0(-1), c10=3(1)
    
    # Step 3: Kronecker product to get 6x6 (rank 49)
    # U6[i*6+j, r1*7+r2] = U2[i*2+j1, r1] * U3[j1*3+j2, r2] where j = j1*3+j2
    # Actually, for Kronecker product of decompositions:
    # If we have A2 (2x2) with U2,V2,W2 and A3 (3x3) with U3,V3,W3
    # Then A6 = A2 ⊗ A3 (6x6) has:
    # U6[(i2*3+i3)*(2*3) + (j2*3+j3), r2*7+r3] = U2[i2*2+j2, r2] * U3[i3*3+j3, r3]
    
    R6 = 7 * 7  # 49
    U6 = np.zeros((36, R6), dtype=float)
    V6 = np.zeros((36, R6), dtype=float)
    W6 = np.zeros((36, R6), dtype=float)
    
    for r2 in range(7):
        for r3 in range(7):
            r6 = r2 * 7 + r3
            for i2 in range(2):
                for j2 in range(2):
                    for i3 in range(3):
                        for j3 in range(3):
                            i6 = i2 * 3 + i3
                            j6 = j2 * 3 + j3
                            U6[i6 * 6 + j6, r6] = U2[i2 * 2 + j2, r2] * U3[i3 * 3 + j3, r3]
                            V6[j6 * 6 + i6, r6] = V2[j2 * 2 + i2, r2] * V3[j3 * 3 + i3, r3]
                            W6[i6 * 6 + j6, r6] = W2[i2 * 2 + j2, r2] * W3[i3 * 3 + j3, r3]
    
    # Step 4: Extract 5x5 from 6x6
    # We keep the first 5x5 block, so we extract indices 0..4 for i,j,k
    # U5[i*5+j, r] for i,j in 0..4
    # V5[j*5+k, r] for j,k in 0..4  
    # W5[k*5+i, r] for k,i in 0..4
    
    U5 = np.zeros((25, R6), dtype=float)
    V5 = np.zeros((25, R6), dtype=float)
    W5 = np.zeros((25, R6), dtype=float)
    
    for r in range(R6):
        for i in range(5):
            for j in range(5):
                # In 6x6, this corresponds to i6=i, j6=j
                U5[i * 5 + j, r] = U6[i * 6 + j, r]
                V5[j * 5 + i, r] = V6[j * 6 + i, r]
                W5[i * 5 + j, r] = W6[i * 6 + j, r]
    
    return {"U": U5.tolist(), "V": V5.tolist(), "W": W5.tolist()}
# EVOLVE-BLOCK-END
