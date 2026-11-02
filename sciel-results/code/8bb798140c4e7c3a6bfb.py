import numpy as np
import time

DIMENSIONS = [2, 3, 3]
GLOBAL_START_TIME = time.time()
TIME_LIMIT = 140.0

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank 17 construction
    # 7 from Strassen for 2x2 core
    # 10 from simple products for the rest
    
    R = 17
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    # Indices for A (2x3):
    # A[0,0] -> U[0]
    # A[0,1] -> U[1]
    # A[0,2] -> U[2]
    # A[1,0] -> U[3]
    # A[1,1] -> U[4]
    # A[1,2] -> U[5]
    
    # Indices for B (3x3):
    # B[0,0] -> V[0]
    # B[0,1] -> V[1]
    # B[0,2] -> V[2]
    # B[1,0] -> V[3]
    # B[1,1] -> V[4]
    # B[1,2] -> V[5]
    # B[2,0] -> V[6]
    # B[2,1] -> V[7]
    # B[2,2] -> V[8]
    
    # Indices for C (2x3) output:
    # C[0,0] -> W[0] (k=0, i=0) -> W[0*2+0] = W[0]
    # C[0,1] -> W[1] (k=1, i=0) -> W[1*2+0] = W[2]
    # C[0,2] -> W[2] (k=2, i=0) -> W[2*2+0] = W[4]
    # C[1,0] -> W[3] (k=0, i=1) -> W[0*2+1] = W[1]
    # C[1,1] -> W[4] (k=1, i=1) -> W[1*2+1] = W[3]
    # C[1,2] -> W[5] (k=2, i=1) -> W[2*2+1] = W[5]
    
    # Wait, W index is k*n + i.
    # n=2.
    # k=0, i=0 -> W[0]
    # k=0, i=1 -> W[1]
    # k=1, i=0 -> W[2]
    # k=1, i=1 -> W[3]
    # k=2, i=0 -> W[4]
    # k=2, i=1 -> W[5]
    
    # Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A11=U[0], A12=U[1], A21=U[3], A22=U[4]
    # B11=V[0], B12=V[1], B21=V[3], B22=V[4]
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C11, C22
    
    # M2 = (A21 + A22)B11
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1 # C21, C22
    
    # M3 = A11(B12 - B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1 # C12, C22
    
    # M4 = A22(B21 - B11)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1 # C11, C21
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1 # C11, C12
    
    # M6 = (A21 - A11)(B11 + B12)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1 # C11, C12
    
    # Check C11: M1 + M4 - M5 + M7 = 1 + 1 - (-1) + 1 = 4? No.
    # C11 = M1 + M4 - M5 + M7
    # W[0,0]=1, W[0,3]=1, W[0,4]=-1, W[0,6]=1. Sum = 1+1-1+1 = 2?
    # Let's re-verify Strassen formulas.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # My W assignments:
    # M1: W[0]=1, W[3]=1
    # M2: W[1]=1, W[3]=-1
    # M3: W[2]=1, W[3]=1
    # M4: W[0]=1, W[1]=1
    # M5: W[0]=-1, W[2]=1
    # M6: W[3]=1
    # M7: W[0]=1, W[2]=1
    
    # C11 (W[0]): 1 (M1) + 1 (M4) - 1 (M5) + 1 (M7) = 2. Should be 1.
    # The Strassen decomposition computes the product.
    # The product is $\sum M_i \times \text{coeff}$.
    # The term $M_1$ contributes $1 \cdot M_1$ to $C_{11}$.
    # So $W[0, 0]$ should be 1.
    # $M_4$ contributes $1 \cdot M_4$ to $C_{11}$.
    # So $W[0, 3]$ should be 1.
    # $M_5$ contributes $-1 \cdot M_5$ to $C_{11}$.
    # So $W[0, 4]$ should be -1.
    # $M_7$ contributes $1 \cdot M_7$ to $C_{11}$.
    # So $W[0, 6]$ should be 1.
    
    # The sum is $1 \cdot M_1 + 1 \cdot M_4 - 1 \cdot M_5 + 1 \cdot M_7$.
    # This is the correct formula for $C_{11}$.
    # So the W values are correct.
    
    # Now the 10 simple products.
    # A13=U[2], A23=U[5]
    # B31=V[6], B32=V[7], B33=V[8]
    # B13=V[2], B23=V[5]
    
    # P8 = A13 B31
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1 # C11
    
    # P9 = A13 B32
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1 # C12
    
    # P10 = A23 B31
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1 # C21
    
    # P11 = A23 B32
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1 # C22
    
    # P12 = A11 B13
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1 # C13
    
    # P13 = A12 B23
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1 # C13
    
    # P14 = A13 B33
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1 # C13
    
    # P15 = A21 B13
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1 # C23
    
    # P16 = A22 B23
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1 # C23
    
    # P17 = A23 B33
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1 # C23
    
    # Verify
    def verify(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for r in range(R):
                        val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook 18
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
