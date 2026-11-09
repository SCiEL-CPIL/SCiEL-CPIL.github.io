import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def _get_strassen_2x2():
    """
    Returns U, V, W for Strassen 2x2.
    Shapes: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen constants
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # U (A coeffs)
    # Row 0: A11, Row 1: A12, Row 2: A21, Row 3: A22 (Row-major 2x2)
    # Actually, index = i*2 + j.
    # A11: 0, A12: 1, A21: 2, A22: 3
    
    # M1: A11(0), A22(3)
    U[0, 0] = 1; U[3, 0] = 1
    # M2: A21(2), A22(3)
    U[2, 1] = 1; U[3, 1] = 1
    # M3: A11(0)
    U[0, 2] = 1
    # M4: A22(3)
    U[3, 3] = 1
    # M5: A11(0), A12(1)
    U[0, 4] = 1; U[1, 4] = 1
    # M6: A21(2), A11(0)
    U[2, 5] = 1; U[0, 5] = -1
    # M7: A12(1), A22(3)
    U[1, 6] = 1; U[3, 6] = -1
    
    # V (B coeffs)
    # B11: 0, B12: 1, B21: 2, B22: 3
    
    # M1: B11(0), B22(3)
    V[0, 0] = 1; V[3, 0] = 1
    # M2: B11(0)
    V[0, 1] = 1
    # M3: B12(1), B22(3)
    V[1, 2] = 1; V[3, 2] = -1
    # M4: B21(2), B11(0)
    V[2, 3] = 1; V[0, 3] = -1
    # M5: B22(3)
    V[3, 4] = 1
    # M6: B11(0), B12(1)
    V[0, 5] = 1; V[1, 5] = 1
    # M7: B21(2), B22(3)
    V[2, 6] = 1; V[3, 6] = 1
    
    # W (C coeffs)
    # C11: 0, C12: 1, C21: 2, C22: 3
    # Trace convention: W[k*2+i]. 
    # C[i,k]. 
    # C11 (i=0,k=0): idx 0
    # C12 (i=0,k=1): idx 1
    # C21 (i=1,k=0): idx 2
    # C22 (i=1,k=1): idx 3
    
    # C11 = M1 - M4 + M6 - M7
    W[0, 0] = 1; W[0, 3] = -1; W[0, 5] = 1; W[0, 6] = -1
    # C12 = M3 + M5
    W[1, 2] = 1; W[1, 4] = 1
    # C21 = M2 + M4
    W[2, 1] = 1; W[2, 3] = 1
    # C22 = M1 - M3 + M2 - M5
    W[3, 0] = 1; W[3, 2] = -1; W[3, 1] = 1; W[3, 4] = -1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 6, 6, 6
    rank = 189
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U2, V2, W2 = _get_strassen_2x2()
    
    # 3x3 grid of 2x2 blocks
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                base_col = (bi * 3 + bj) * 3 + bk
                col_off = base_col * 7
                
                for r in range(7):
                    # A part
                    for i in range(2):
                        for j in range(2):
                            u_row = (2*bi + i) * 6 + (2*bj + j)
                            U[u_row, col_off + r] += U2[i*2+j, r]
                            
                    # B part
                    for j in range(2):
                        for k in range(2):
                            v_row = (2*bj + j) * 6 + (2*bk + k)
                            V[v_row, col_off + r] += V2[j*2+k, r]
                            
                    # C part
                    # W index: k*6 + i
                    for i in range(2):
                        for k in range(2):
                            w_row = (2*bk + k) * 6 + (2*bi + i)
                            W[w_row, col_off + r] += W2[k*2+i, r]
                            
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
