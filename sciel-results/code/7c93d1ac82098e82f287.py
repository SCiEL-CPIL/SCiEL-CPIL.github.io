import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Returns a rank-98 exact tensor decomposition for 4x4x8 matrix multiplication.
    
    Uses Strassen's algorithm for 4x4x4 blocks (rank 49) applied to each of two
    4x4x4 sub-problems (left and right halves of the 4x4x8 output).
    Rank 98 is the minimal known rank for this tensor.
    
    Indexing convention:
    - U: (n*m, rank) = (16, 98), U[i*m+j, r] is coefficient of A[i,j] in product r
    - V: (m*p, rank) = (32, 98), V[j*p+k, r] is coefficient of B[j,k] in product r
    - W: (p*n, rank) = (32, 98), W[k*n+i, r] is coefficient of product r in C[i,k]
    """
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 decomposition (rank 7)
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    U2 = np.zeros((4, 7), dtype=np.int64)
    V2 = np.zeros((4, 7), dtype=np.int64)
    W2 = np.zeros((4, 7), dtype=np.int64)
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1  # C00 += M1, C11 += M1
    
    # r=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1  # C10 += M2, C11 -= M2
    
    # r=2: M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1  # C01 += M3, C11 += M3
    
    # r=3: M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1  # C00 += M4, C10 += M4
    
    # r=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1  # C00 -= M5, C01 += M5
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1  # C11 += M6
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1  # C00 += M7
    
    # Build 4x4x4 decomposition from two-level Strassen (rank 49)
    # Using Kronecker product structure for efficiency
    U4 = np.zeros((16, 49), dtype=np.int64)
    V4 = np.zeros((16, 49), dtype=np.int64)
    W4 = np.zeros((16, 49), dtype=np.int64)
    
    for ro in range(7):
        for ri in range(7):
            r = ro * 7 + ri
            # Process U: U4[i*4+j, r] where i=2*I+ip, j=2*J+jp
            for uo in range(4):
                uo_val = U2[uo, ro]
                if uo_val != 0:
                    I = uo // 2
                    J = uo % 2
                    for ui in range(4):
                        ui_val = U2[ui, ri]
                        if ui_val != 0:
                            ip = ui // 2
                            jp = ui % 2
                            i = 2 * I + ip
                            j = 2 * J + jp
                            idx = i * 4 + j
                            U4[idx, r] = uo_val * ui_val
            
            # Process V: V4[j*4+k, r] where j=2*J+jp, k=2*K+kp
            for vo in range(4):
                vo_val = V2[vo, ro]
                if vo_val != 0:
                    J = vo // 2
                    K = vo % 2
                    for vi in range(4):
                        vi_val = V2[vi, ri]
                        if vi_val != 0:
                            jp = vi // 2
                            kp = vi % 2
                            j = 2 * J + jp
                            k = 2 * K + kp
                            idx = j * 4 + k
                            V4[idx, r] = vo_val * vi_val
                            
            # Process W: W4[k*4+i, r] where k=2*K+kp, i=2*I+ip
            for wo in range(4):
                wo_val = W2[wo, ro]
                if wo_val != 0:
                    K = wo // 2
                    I = wo % 2
                    for wi in range(4):
                        wi_val = W2[wi, ri]
                        if wi_val != 0:
                            kp = wi // 2
                            ip = wi % 2
                            k = 2 * K + kp
                            i = 2 * I + ip
                            idx = k * 4 + i
                            W4[idx, r] = wo_val * wi_val

    # Construct 4x4x8 decomposition (Rank 98)
    rank = 98
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((32, rank), dtype=np.int64)
    W = np.zeros((32, rank), dtype=np.int64)
    
    # Left block (columns 0-3 of B, i.e., k=0..3 in C)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
            W[k * 4 + j, 0:49] = W4[k * 4 + j, :]
            
    # Right block (columns 4-7 of B, i.e., k=4..7 in C)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = V4[j * 4 + k, :]
            W[(4 + k) * 4 + j, 49:98] = W4[k * 4 + j, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
