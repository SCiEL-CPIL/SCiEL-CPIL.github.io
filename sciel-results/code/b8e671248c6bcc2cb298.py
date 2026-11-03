import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    # Strassen 2x2 decomposition factors
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    # Indices:
    # U row: i*2 + j for A[i,j]
    # V row: j*2 + k for B[j,k]
    # W row: k*2 + i for C[k,i] (trace convention W[k*n+i])
    
    U_S = np.zeros((4, 7), dtype=int)
    V_S = np.zeros((4, 7), dtype=int)
    W_S = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11) -> C00 + C11
    # A00: idx 0, A11: idx 3
    # B00: idx 0, B11: idx 3
    # C00: k=0,i=0 -> idx 0. C11: k=1,i=1 -> idx 3
    U_S[0, 0] = 1; U_S[3, 0] = 1
    V_S[0, 0] = 1; V_S[3, 0] = 1
    W_S[0, 0] = 1; W_S[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10
    # A10: idx 2, A11: idx 3
    # B00: idx 0
    # C10: k=1,i=0 -> idx 2
    U_S[2, 1] = 1; U_S[3, 1] = 1
    V_S[0, 1] = 1
    W_S[2, 1] = 1
    
    # M3: A00(B01-B11) -> C01
    # A00: idx 0
    # B01: idx 1, B11: idx 3
    # C01: k=0,i=1 -> idx 1
    U_S[0, 2] = 1
    V_S[1, 2] = 1; V_S[3, 2] = -1
    W_S[1, 2] = 1
    
    # M4: A11(B10-B00) -> C10
    # A11: idx 3
    # B10: idx 2, B00: idx 0
    # C10: k=1,i=0 -> idx 2
    U_S[3, 3] = 1
    V_S[2, 3] = 1; V_S[0, 3] = -1
    W_S[2, 3] = 1
    
    # M5: (A00+A10)B11 -> -C00 + C11
    # A00: idx 0, A10: idx 2
    # B11: idx 3
    # C00: idx 0, C11: idx 3
    U_S[0, 4] = 1; U_S[2, 4] = 1
    V_S[3, 4] = 1
    W_S[0, 4] = -1; W_S[3, 4] = 1
    
    # M6: (A01-A00)(B00+B10) -> C11
    # A01: idx 1, A00: idx 0
    # B00: idx 0, B10: idx 2
    # C11: idx 3
    U_S[1, 5] = 1; U_S[0, 5] = -1
    V_S[0, 5] = 1; V_S[2, 5] = 1
    W_S[3, 5] = 1
    
    # M7: (A10-A11)(B11+B01) -> C00
    # A10: idx 2, A11: idx 3
    # B11: idx 3, B01: idx 1
    # C00: idx 0
    U_S[2, 6] = 1; U_S[3, 6] = -1
    V_S[3, 6] = 1; V_S[1, 6] = 1
    W_S[0, 6] = 1

    # Kronecker product for 4x4
    # U = U_S (x) U_S
    # Row index R = IA * 4 + IB
    # Col index C = CA * 7 + CB
    # U[R, C] = U_S[IA, CA] * U_S[IB, CB]
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for IA in range(4):
        for IB in range(4):
            R = IA * 4 + IB
            for CA in range(7):
                for CB in range(7):
                    C = CA * 7 + CB
                    U[R, C] = U_S[IA, CA] * U_S[IB, CB]
                    V[R, C] = V_S[IA, CA] * V_S[IB, CB]
                    W[R, C] = W_S[IA, CA] * W_S[IB, CB]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
