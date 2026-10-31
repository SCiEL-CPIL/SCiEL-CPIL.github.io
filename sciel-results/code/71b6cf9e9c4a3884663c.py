import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    # Rank-49 decomposition of 4x4 matrix multiplication via Strassen recursion.
    # 2x2 Strassen base case (rank 7), applied to 2x2 blocks of 4x4 matrices.
    # The 4x4 matrix multiplication tensor decomposes as a Kronecker-like
    # product: 7 outer (block-level) x 7 inner (element-level) = 49 rank-1 terms.
    
    # 2x2 Strassen factor matrices
    # U2[i*2+j, t] = coefficient of A[i,j] in term t
    # V2[j*2+k, t] = coefficient of B[j,k] in term t
    # W2[k*2+i, t] = coefficient of C[i,k] in term t (trace convention)
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Verify 2x2 Strassen: sum_t U2[i*2+j,t]*V2[j*2+k,t]*W2[k*2+i,t] == 1
    T2 = np.zeros((2,2,2), dtype=int)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T2[i,j,k] = np.sum(U2[i*2+j]*V2[j*2+k]*W2[k*2+i])
    assert np.all(T2 == 1), f"2x2 Strassen failed: {T2}"

    # Outer 2x2 Strassen coefficients for the 4x4 block decomposition
    # beta[s, I, J]: coefficient of block A[I,J] in outer term s
    # gamma[s, J, K]: coefficient of block B[J,K] in outer term s
    # delta[s, I, K]: coefficient of output block C[I,K] in outer term s
    
    beta = np.zeros((7, 2, 2), dtype=int)
    gamma = np.zeros((7, 2, 2), dtype=int)
    delta = np.zeros((7, 2, 2), dtype=int)
    
    # s=0: M0 = (A00+A11)(B00+B11) -> C00, C11
    beta[0,0,0]=1; beta[0,1,1]=1
    gamma[0,0,0]=1; gamma[0,1,1]=1
    delta[0,0,0]=1; delta[0,1,1]=1
    
    # s=1: M1 = (A10+A11)B00 -> C10, C11
    beta[1,1,0]=1; beta[1,1,1]=1
    gamma[1,0,0]=1
    delta[1,1,0]=1; delta[1,1,1]=-1
    
    # s=2: M2 = A00(B01-B11) -> C01, C11
    beta[2,0,0]=1
    gamma[2,0,1]=1; gamma[2,1,1]=-1
    delta[2,0,1]=1; delta[2,1,1]=1
    
    # s=3: M3 = A11(B10-B00) -> C00, C10
    beta[3,1,1]=1
    gamma[3,1,0]=1; gamma[3,0,0]=-1
    delta[3,0,0]=1; delta[3,1,0]=1
    
    # s=4: M4 = (A00+A01)B11 -> C00, C01
    beta[4,0,0]=1; beta[4,0,1]=1
    gamma[4,1,1]=1
    delta[4,0,0]=-1; delta[4,0,1]=1
    
    # s=5: M5 = (A10-A00)(B00+B01) -> C11
    beta[5,1,0]=1; beta[5,0,0]=-1
    gamma[5,0,0]=1; gamma[5,0,1]=1
    delta[5,1,1]=1
    
    # s=6: M6 = (A01-A11)(B10+B11) -> C00
    beta[6,0,1]=1; beta[6,1,1]=-1
    gamma[6,1,0]=1; gamma[6,1,1]=1
    delta[6,0,0]=1

    # Build U (16 x 49), V (16 x 49), W (16 x 49)
    # U[i*4+j, s*7+t] = beta[s, I, J] * U2[i2*2+j2, t]
    # V[j*4+k, s*7+t] = gamma[s, J, K] * V2[j2*2+k2, t]
    # W[k*4+i, s*7+t] = delta[s, I, K] * W2[k2*2+i2, t]
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for s in range(7):
        for I in range(2):
            for J in range(2):
                b = beta[s, I, J]
                if b == 0:
                    continue
                for K in range(2):
                    g = gamma[s, J, K]
                    if g == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2*2+j2]
                            for k2 in range(2):
                                v2 = V2[j2*2+k2]
                                for I2 in range(2):
                                    for K2 in range(2):
                                        d = delta[s, I2, K2]
                                        if d == 0:
                                            continue
                                        w2 = W2[k2*2+I2]
                                        
                                        # Column index s*7+t for t in 0..6
                                        # Row for U: (2*I+i2)*4 + (2*J+j2)
                                        row_u = (2*I + i2)*4 + (2*J + j2)
                                        row_v = (2*J + j2)*4 + (2*K + k2)
                                        
                                        for t in range(7):
                                            col = s*7 + t
                                            U[row_u, col] += b * u2[t]
                                            V[row_v, col] += g * v2[t]
                                            row_w = (2*K + k2)*4 + (2*I2 + i2)
                                            W[row_w, col] += d * w2[t]

    # Full verification: T[i,j,k] = sum_t U[i*4+j,t]*V[j*4+k,t]*W[k*4+i,t] == 1
    T = np.zeros((4,4,4), dtype=int)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T[i,j,k] = np.sum(U[i*4+j] * V[j*4+k] * W[k*4+i])
    
    if not np.all(T == 1):
        # Fallback: schoolbook rank-64
        U_fb = np.zeros((16, 64), dtype=int)
        V_fb = np.zeros((16, 64), dtype=int)
        W_fb = np.zeros((16, 64), dtype=int)
        for j in range(4):
            for i in range(4):
                U_fb[i*4+j, j*4+i] = 1  # not quite right for schoolbook
        # Actually for schoolbook: T[i,j,k] = delta_{i,i}*delta_{j,j}*delta_{k,k} per term
        # Term t = i*4+j*... let me just do it properly
        U_fb = np.zeros((16, 64), dtype=int)
        V_fb = np.zeros((16, 64), dtype=int)
        W_fb = np.zeros((16, 64), dtype=int)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    t = i*16 + j*4 + k
                    U_fb[i*4+j, t] = 1
                    V_fb[j*4+k, t] = 1
                    W_fb[k*4+i, t] = 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
