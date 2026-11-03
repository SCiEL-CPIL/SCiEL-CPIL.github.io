"""Matrix multiplication 6x6x6 via Kronecker product of Winograd 3x3 (rank 8) and Strassen 2x2 (rank 4)."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start = time.time()
    
    # Winograd 3x3 rank-8 decomposition
    A3 = np.zeros((9, 8), dtype=float)
    B3 = np.zeros((9, 8), dtype=float)
    C3 = np.zeros((9, 8), dtype=float)
    
    # p0 = (a00+a10+a20)(b00+b01+b02)
    A3[0,0]=1; A3[3,0]=1; A3[6,0]=1
    B3[0,0]=1; B3[1,0]=1; B3[2,0]=1
    C3[0,0]=1; C3[3,0]=1; C3[6,0]=1
    
    # p1 = (a10+a20)(b01+b02)
    A3[3,1]=1; A3[6,1]=1
    B3[1,1]=1; B3[2,1]=1
    C3[0,1]=-1; C3[3,1]=-1; C3[6,1]=-1
    C3[1,1]=1; C3[4,1]=1; C3[7,1]=1
    C3[2,1]=1; C3[5,1]=1; C3[8,1]=1
    
    # p2 = (a00+a10)(b00+b01)
    A3[0,2]=1; A3[3,2]=1
    B3[0,2]=1; B3[1,2]=1
    C3[0,2]=-1; C3[3,2]=-1
    C3[1,2]=1; C3[4,2]=1
    
    # p3 = (a00+a20)(b00+b02)
    A3[0,3]=1; A3[6,3]=1
    B3[0,3]=1; B3[2,3]=1
    C3[0,3]=1; C3[6,3]=1
    C3[2,3]=1; C3[8,3]=1
    
    # p4 = a01(b10-b12)
    A3[1,4]=1
    B3[3,4]=1; B3[5,4]=-1
    C3[3,4]=-1; C3[6,4]=1
    
    # p5 = a11(b11-b10)
    A3[4,5]=1
    B3[4,5]=1; B3[3,5]=-1
    C3[3,5]=1; C3[6,5]=-1
    C3[4,5]=-1; C3[7,5]=1
    
    # p6 = a21(b12-b11)
    A3[7,6]=1
    B3[5,6]=1; B3[4,6]=-1
    C3[6,6]=1; C3[3,6]=-1
    C3[7,6]=-1; C3[4,6]=1
    
    # p7 = a12(b21-b20)
    A3[5,7]=1
    B3[7,7]=1; B3[6,7]=-1
    C3[6,7]=1; C3[3,7]=-1
    C3[8,7]=-1; C3[5,7]=1
    
    # Strassen 2x2 rank-4
    A2 = np.zeros((4, 4), dtype=float)
    B2 = np.zeros((4, 4), dtype=float)
    C2 = np.zeros((4, 4), dtype=float)
    
    # m0 = (a00+a11)(b00+b11)
    A2[0,0]=1; A2[3,0]=1
    B2[0,0]=1; B2[3,0]=1
    C2[0,0]=1; C2[1,0]=1; C2[2,0]=1; C2[3,0]=1
    
    # m1 = (a10+a11)(b00)
    A2[2,1]=1; A2[3,1]=1
    B2[0,1]=1
    C2[0,1]=-1; C2[2,1]=1; C2[3,1]=-1
    
    # m2 = a00(b01-b11)
    A2[0,2]=1
    B2[1,2]=1; B2[3,2]=-1
    C2[0,2]=-1; C2[1,2]=1
    
    # m3 = a11(b10-b00)
    A2[3,3]=1
    B2[2,3]=1; B2[0,3]=-1
    C2[0,3]=1; C2[2,3]=1
    
    # Build permutation for row ordering
    # 6x6 index (i,j) -> 3x2 decomposition: i=2*i3+i2, j=2*j3+j2
    # Kronecker row index for A: (i3*3+j3)*4 + (i2*2+j2)
    # Desired row index: i*6+j = (2*i3+i2)*6 + (2*j3+j2)
    
    perm_A = np.zeros(36, dtype=int)
    perm_B = np.zeros(36, dtype=int)
    perm_C = np.zeros(36, dtype=int)
    
    for i3 in range(3):
        for j3 in range(3):
            for i2 in range(2):
                for j2 in range(2):
                    i = 2*i3 + i2
                    j = 2*j3 + j2
                    # A index: i*6+j
                    a_idx = i*6 + j
                    # Kronecker index: (i3*3+j3)*4 + (i2*2+j2)
                    k_idx = (i3*3+j3)*4 + (i2*2+j2)
                    perm_A[a_idx] = k_idx
    
    for j3 in range(3):
        for k3 in range(3):
            for j2 in range(2):
                for k2 in range(2):
                    j = 2*j3 + j2
                    k = 2*k3 + k2
                    # B index: j*6+k
                    b_idx = j*6 + k
                    # Kronecker index: (j3*3+k3)*4 + (j2*2+k2)
                    k_idx = (j3*3+k3)*4 + (j2*2+k2)
                    perm_B[b_idx] = k_idx
    
    for k3 in range(3):
        for i3 in range(3):
            for k2 in range(2):
                for i2 in range(2):
                    k = 2*k3 + k2
                    i = 2*i3 + i2
                    # C index (trace convention): k*6+i
                    c_idx = k*6 + i
                    # Kronecker index: (k3*3+i3)*4 + (k2*2+i2)
                    k_idx = (k3*3+i3)*4 + (k2*2+i2)
                    perm_C[c_idx] = k_idx
    
    # Build permuted matrices
    U = np.zeros((36, 32), dtype=float)
    V = np.zeros((36, 32), dtype=float)
    W = np.zeros((36, 32), dtype=float)
    
    U_kron = np.kron(A3, A2)
    V_kron = np.kron(B3, B2)
    W_kron = np.kron(C3, C2)
    
    for desired in range(36):
        U[desired, :] = U_kron[perm_A[desired], :]
        V[desired, :] = V_kron[perm_B[desired], :]
        W[desired, :] = W_kron[perm_C[desired], :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
