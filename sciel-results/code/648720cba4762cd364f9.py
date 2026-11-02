import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    # Strassen decomposition for 2x2x2
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    U_S = np.zeros((4, 7), dtype=int)
    V_S = np.zeros((4, 7), dtype=int)
    W_S = np.zeros((4, 7), dtype=int)
    
    # M1 = (A11 + A22)(B11 + B22)
    U_S[0, 0] = 1 # A11
    U_S[3, 0] = 1 # A22
    V_S[0, 0] = 1 # B11
    V_S[3, 0] = 1 # B22
    W_S[0, 0] = 1 # C11
    W_S[3, 0] = 1 # C22
    
    # M2 = (A21 + A22)B11
    U_S[2, 1] = 1 # A21
    U_S[3, 1] = 1 # A22
    V_S[0, 1] = 1 # B11
    W_S[2, 1] = 1 # C21
    
    # M3 = A11(B12 - B22)
    U_S[0, 2] = 1 # A11
    V_S[1, 2] = 1 # B12
    V_S[3, 2] = -1 # B22
    W_S[1, 2] = 1 # C12
    W_S[3, 2] = 1 # C22
    
    # M4 = A22(B21 - B11)
    U_S[3, 3] = 1 # A22
    V_S[2, 3] = 1 # B21
    V_S[0, 3] = -1 # B11
    W_S[0, 3] = 1 # C11
    W_S[2, 3] = 1 # C21
    
    # M5 = (A11 + A12)B22
    U_S[0, 4] = 1 # A11
    U_S[1, 4] = 1 # A12
    V_S[3, 4] = 1 # B22
    W_S[0, 4] = -1 # C11
    W_S[1, 4] = 1 # C12
    
    # M6 = (A21 - A11)(B11 + B12)
    U_S[2, 5] = 1 # A21
    U_S[0, 5] = -1 # A11
    V_S[0, 5] = 1 # B11
    V_S[1, 5] = 1 # B12
    W_S[3, 5] = 1 # C22
    
    # M7 = (A12 - A22)(B21 + B22)
    U_S[1, 6] = 1 # A12
    U_S[3, 6] = -1 # A22
    V_S[2, 6] = 1 # B21
    V_S[3, 6] = 1 # B22
    W_S[0, 6] = 1 # C11
    W_S[1, 6] = 1 # C12
    
    # Schoolbook decomposition for 1x2x4
    # U_SB: 2x8, V_SB: 8x8, W_SB: 4x8
    U_SB = np.zeros((2, 8), dtype=int)
    V_SB = np.zeros((8, 8), dtype=int)
    W_SB = np.zeros((4, 8), dtype=int)
    
    for j in range(2):
        for k in range(4):
            c = j * 4 + k
            U_SB[j, c] = 1
            V_SB[j * 4 + k, c] = 1
            W_SB[k, c] = 1
            
    # Combine using Kronecker product
    # U: 8x56
    U = np.kron(U_S, U_SB)
    # V: 32x56
    V = np.kron(V_S, V_SB)
    # W: 16x56
    W = np.kron(W_S, W_SB)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
