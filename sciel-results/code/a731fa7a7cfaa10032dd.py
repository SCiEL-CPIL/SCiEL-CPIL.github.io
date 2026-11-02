"""Matrix multiplication 4x4x6 via Kronecker product decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6

    # Strategy: 4x4x6 = (2x2x2) ⊗ (2x2x3)
    # Factor 1: 2x2x2 at rank 7 (Strassen)
    # Factor 2: 2x2x3 at rank 11 (Strassen for 2 cols + 4 trivial for 3rd col)
    # Total rank = 7 * 11 = 77

    # --- Strassen for 2x2x2 ---
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]], C = A@B
    # U1[i*2+j, c], V1[j*2+k, c], W1[k*2+i, c]
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)

    # M1 = (a00+a11)(b00+b11)
    U1[0,0]=1; U1[3,0]=1; V1[0,0]=1; V1[3,0]=1; W1[0,0]=1; W1[3,0]=1
    # M2 = (a10+a11)b00
    U1[2,1]=1; U1[3,1]=1; V1[0,1]=1; W1[1,1]=1; W1[3,1]=-1
    # M3 = a00(b01-b11)
    U1[0,2]=1; V1[1,2]=1; V1[3,2]=-1; W1[2,2]=1; W1[3,2]=1
    # M4 = a11(b10-b00)
    U1[3,3]=1; V1[2,3]=1; V1[0,3]=-1; W1[0,3]=1; W1[1,3]=1
    # M5 = (a00+a01)b11
    U1[0,4]=1; U1[1,4]=1; V1[3,4]=1; W1[0,4]=-1; W1[2,4]=1
    # M6 = (a10-a00)(b00+b01)
    U1[2,5]=1; U1[0,5]=-1; V1[0,5]=1; V1[1,5]=1; W1[3,5]=1
    # M7 = (a01-a11)(b10+b11)
    U1[1,6]=1; U1[3,6]=-1; V1[2,6]=1; V1[3,6]=1; W1[0,6]=1

    # --- 2x2x3 decomposition ---
    # A is 2x2, B is 2x3, C is 2x3
    # Use Strassen (7 terms) for C[:,0:2] and 4 trivial terms for C[:,2]
    U2 = np.zeros((4, 11), dtype=float)
    V2 = np.zeros((6, 11), dtype=float)  # 2*3=6 rows
    W2 = np.zeros((6, 11), dtype=float)  # 3*2=6 rows (k*2+i)

    # Copy Strassen part for columns 0-1
    U2[:, :7] = U1
    # V2 rows: j*3+k for j in 0..1, k in 0..2
    # V1 rows: j*2+k for j in 0..1, k in 0..1
    # Map: V1[j*2+k] -> V2[j*3+k] for k=0,1; V2[j*3+2]=0
    for j in range(2):
        for k in range(2):
            V2[j*3+k, :7] = V1[j*2+k, :7]
    V2[2:4, :7] = 0  # row 2,3 (j=0,k=2 and j=1,k=2) are 0
    # Actually: V2 row index = j*3+k
    # j=0: rows 0,1,2 (k=0,1,2)
    # j=1: rows 3,4,5 (k=0,1,2)
    V2[0,:7]=V1[0,:7]; V2[1,:7]=V1[1,:7]; V2[2,:7]=0
    V2[3,:7]=V1[2,:7]; V2[4,:7]=V1[3,:7]; V2[5,:7]=0

    # W2 rows: k*2+i for k in 0..2, i in 0..1
    # For Strassen part (columns 0-1 of C), W2[k*2+i] = W1[k*2+i] for k=0,1
    for k in range(2):
        for i in range(2):
            W2[k*2+i, :7] = W1[k*2+i, :7]
    W2[4:6, :7] = 0  # k=2 rows are 0

    # Trivial part for column 2: 4 terms (indices 7-10)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    U2[0,7]=1; V2[2,7]=1; W2[4,7]=1   # A[0,0]*B[0,2] -> C[0,2] (W2 row: k=2,i=0 => row 4)
    U2[1,8]=1; V2[5,8]=1; W2[4,8]=1   # A[0,1]*B[1,2] -> C[0,2]
    U2[2,9]=1; V2[2,9]=1; W2[5,9]=1   # A[1,0]*B[0,2] -> C[1,2] (W2 row: k=2,i=1 => row 5)
    U2[3,10]=1; V2[5,10]=1; W2[5,10]=1 # A[1,1]*B[1,2] -> C[1,2]

    # --- Kronecker product construction ---
    # A(4x4) = A1(2x2) ⊗ A2(2x2): A[i,j] where i=i1*2+i2, j=j1*2+j2
    # B(4x6) = B1(2x2) ⊗ B2(2x3): B[j,k] where j=j1*2+j2, k=k1*3+k2
    # C(4x6) = C1(2x2) ⊗ C2(2x3): C[i,k] where i=i1*2+i2, k=k1*3+k2

    rank = 7 * 11
    U = np.zeros((16, rank), dtype=float)  # 4*4=16
    V = np.zeros((24, rank), dtype=float)  # 4*6=24
    W = np.zeros((24, rank), dtype=float)  # 6*4=24

    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            # U: row = i*4+j = (i1*2+i2)*4 + (j1*2+j2)
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            row = (i1*2+i2)*4 + (j1*2+j2)
                            U[row, c] = U1[i1*2+j1, c1] * U2[i2*2+j2, c2]

            # V: row = j*6+k = (j1*2+j2)*6 + (k1*3+k2)
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            row = (j1*2+j2)*6 + (k1*3+k2)
                            V[row, c] = V1[j1*2+k1, c1] * V2[j2*3+k2, c2]

            # W: row = k*4+i = (k1*3+k2)*4 + (i1*2+i2)
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            row = (k1*3+k2)*4 + (i1*2+i2)
                            W[row, c] = W1[k1*2+i1, c1] * W2[k2*2+i2, c2]

    # Convert to lists of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
