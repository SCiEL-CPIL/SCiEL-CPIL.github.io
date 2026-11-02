"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    n2, m2, p2 = 2, 2, 2
    r2 = 7
    U2 = np.zeros((n2*m2, r2))
    V2 = np.zeros((m2*p2, r2))
    W2 = np.zeros((p2*n2, r2))
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]]
    # C = [[C00,C01],[C10,C11]]
    #
    # M1 = A00*(B00+B11) -> C01+=1, C11+=1
    # M2 = (A00+A01)*B11 -> C01+=1, C11-=1
    # M3 = A10*(B00-B10) -> C10+=1
    # M4 = A11*(B10+B11) -> C00+=1, C11+=1
    # M5 = (A00+A11)*(B00+B11) -> C00+=1, C10+=1
    # M6 = (A00-A10)*(B00+B10) -> C11+=1
    # M7 = (A01-A11)*(B10+B11) -> C00-=1
    
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # A indices: i (row 0-1), j (col 0-1) -> U[i*2+j]
    # B indices: j (row 0-1), k (col 0-1) -> V[j*2+k]
    # C indices: i (row 0-1), k (col 0-1) -> W[k*2+i]
    
    # c=0: M1 = A[0,0] * (B[0,0] + B[1,1])
    U2[0, 0] = 1   # A[0,0]
    V2[0, 0] = 1   # B[0,0]
    V2[3, 0] = 1   # B[1,1]
    W2[1, 0] = 1   # C[0,1]: k=0,i=1 -> W[0*2+1]=W[1]
    W2[3, 0] = 1   # C[1,1]: k=1,i=1 -> W[1*2+1]=W[3]
    
    # c=1: M2 = (A[0,0]+A[0,1]) * B[1,1]
    U2[0, 1] = 1   # A[0,0]
    U2[1, 1] = 1   # A[0,1]
    V2[3, 1] = 1   # B[1,1]
    W2[1, 1] = 1   # C[0,1]
    W2[3, 1] = -1  # C[1,1]
    
    # c=2: M3 = A[1,0] * (B[0,0] - B[1,0])
    U2[2, 2] = 1   # A[1,0]
    V2[0, 2] = 1   # B[0,0]
    V2[2, 2] = -1  # B[1,0]
    W2[2, 2] = 1   # C[1,0]: k=1,i=0 -> W[1*2+0]=W[2]
    
    # c=3: M4 = A[1,1] * (B[1,0] + B[1,1])
    U2[3, 3] = 1   # A[1,1]
    V2[2, 3] = 1   # B[1,0]
    V2[3, 3] = 1   # B[1,1]
    W2[0, 3] = 1   # C[0,0]: k=0,i=0 -> W[0*2+0]=W[0]
    W2[3, 3] = 1   # C[1,1]
    
    # c=4: M5 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    U2[0, 4] = 1   # A[0,0]
    U2[3, 4] = 1   # A[1,1]
    V2[0, 4] = 1   # B[0,0]
    V2[3, 4] = 1   # B[1,1]
    W2[0, 4] = 1   # C[0,0]
    W2[2, 4] = 1   # C[1,0]
    
    # c=5: M6 = (A[0,0]-A[1,0]) * (B[0,0]+B[1,0])
    U2[0, 5] = 1   # A[0,0]
    U2[2, 5] = -1  # A[1,0]
    V2[0, 5] = 1   # B[0,0]
    V2[2, 5] = 1   # B[1,0]
    W2[3, 5] = 1   # C[1,1]
    
    # c=6: M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U2[1, 6] = 1   # A[0,1]
    U2[3, 6] = -1  # A[1,1]
    V2[2, 6] = 1   # B[1,0]
    V2[3, 6] = 1   # B[1,1]
    W2[0, 6] = -1  # C[0,0]
    
    # Schoolbook decomposition for 3x3 matrix multiplication (rank 27)
    n3, m3, p3 = 3, 3, 3
    r3 = 27
    U3 = np.zeros((n3*m3, r3))
    V3 = np.zeros((m3*p3, r3))
    W3 = np.zeros((p3*n3, r3))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j, col] = 1
                V3[j*3+k, col] = 1
                W3[k*3+i, col] = 1
                col += 1
    
    # Kronecker product construction: T_6 = T_2 ⊗ T_3
    # Total rank = 7 * 27 = 189
    r_total = r2 * r3
    U6 = np.zeros((36, r_total))
    V6 = np.zeros((36, r_total))
    W6 = np.zeros((36, r_total))
    
    for c2 in range(r2):
        for c3 in range(r3):
            c6 = c2 * r3 + c3
            # U6[i*6+j, c6] = U2[i2*2+j2, c2] * U3[i3*3+j3, c3]
            # where i = 2*i2 + i3, j = 2*j2 + j3
            for i2 in range(2):
                for j2 in range(2):
                    for i3 in range(3):
                        for j3 in range(3):
                            i = 2*i2 + i3
                            j = 2*j2 + j3
                            U6[i*6+j, c6] = U2[i2*2+j2, c2] * U3[i3*3+j3, c3]
            # V6[j*6+k, c6] = V2[j2*2+k2, c2] * V3[j3*3+k3, c3]
            # where j = 2*j2 + j3, k = 2*k2 + k3
            for j2 in range(2):
                for k2 in range(2):
                    for j3 in range(3):
                        for k3 in range(3):
                            j = 2*j2 + j3
                            k = 2*k2 + k3
                            V6[j*6+k, c6] = V2[j2*2+k2, c2] * V3[j3*3+k3, c3]
            # W6[k*6+i, c6] = W2[k2*2+i2, c2] * W3[k3*3+i3, c3]
            # where k = 2*k2 + k3, i = 2*i2 + i3
            for k2 in range(2):
                for i2 in range(2):
                    for k3 in range(3):
                        for i3 in range(3):
                            k = 2*k2 + k3
                            i = 2*i2 + i3
                            W6[k*6+i, c6] = W2[k2*2+i2, c2] * W3[k3*3+i3, c3]
    
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
