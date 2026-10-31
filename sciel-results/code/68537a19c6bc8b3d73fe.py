import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS
    rank = 49

    # 2x2 Strassen factors (rank 7)
    # U2[i, s] for i in 0..3 (i = 2*I + i'), s in 0..6
    # V2[j, s] for j in 0..3 (j = 2*J + j'), s in 0..6
    # W2[k, s] for k in 0..3 (k = 2*K + k'), s in 0..6
    # where the 2x2 indices are I,i' for A, J,j' for B, K,k' for C
    # C[I,K] = sum_s A[I,J]*B[J,K] ... 
    # Actually for 2x2: C[I,K] = sum_s U2[I,s] * V2[K,s] ... no
    # The tensor is T[i,j,k] = delta_{i,k} (for matmul A*B where C[i,k] = sum_j A[i,j]B[j,k])
    # Rank decomposition: T[i,j,k] = sum_s U[i,s] V[j,s] W[k,s]
    # For 2x2: C[i,k] = sum_j A[i,j] B[j,k] = sum_s (sum_j U[i,s] A[i,j]) ... no
    # T[i,j,k] = 1 if i==k else 0
    # U[i,s] V[j,s] W[k,s] summed over s should give delta_{i,k}
    
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U[i,s]: coefficient of A[i,j] in M_s... but U depends on i and j
    # Actually U is (n*m, rank) = (4, 7) for 2x2
    # U[i*2+j, s] = coefficient of A[i,j] in the s-th product
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # s=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1  # A00, A11
    V2[0, 0] = 1; V2[3, 0] = 1  # B00, B11
    W2[0, 0] = 1; W2[3, 0] = 1  # C00, C11
    
    # s=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1  # A10, A11
    V2[0, 1] = 1  # B00
    W2[1, 1] = 1; W2[3, 1] = -1  # C10, C11
    
    # s=2: M3 = A00(B01-B11)
    U2[0, 2] = 1  # A00
    V2[1, 2] = 1; V2[3, 2] = -1  # B01, B11
    W2[2, 2] = 1; W2[3, 2] = 1  # C01, C11
    
    # s=3: M4 = A11(B10-B00)
    U2[3, 3] = 1  # A11
    V2[2, 3] = 1; V2[0, 3] = -1  # B10, B00
    W2[0, 3] = 1; W2[1, 3] = 1  # C00, C10
    
    # s=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1  # A00, A01
    V2[3, 4] = 1  # B11
    W2[0, 4] = -1; W2[2, 4] = 1  # C00, C01
    
    # s=5: M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1  # A10, A00
    V2[0, 5] = 1; V2[1, 5] = 1  # B00, B01
    W2[3, 5] = 1  # C11
    
    # s=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1  # A01, A11
    V2[2, 6] = 1; V2[3, 6] = 1  # B10, B11
    W2[0, 6] = 1  # C00
    
    # Build 4x4 factors via Kronecker product
    # U4[i*4+j, s*7+t] = U2[i, s] * U2[j, t]
    # V4[j*4+k, s*7+t] = V2[j, s] * V2[k, t]
    # W4[k*4+i, s*7+t] = W2[k, s] * W2[i, t]
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for s in range(7):
        for t in range(7):
            c = s * 7 + t
            for i in range(4):
                for j in range(4):
                    U[i * 4 + j, c] = U2[i, s] * U2[j, t]
            for j in range(4):
                for k in range(4):
                    V[j * 4 + k, c] = V2[j, s] * V2[k, t]
            for k in range(4):
                for i in range(4):
                    W[k * 4 + i, c] = W2[k, s] * W2[i, t]
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
