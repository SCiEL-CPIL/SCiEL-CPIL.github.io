import numpy as np

DIMENSIONS = [3, 5, 5]

def schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def strassen_222():
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # M1: (A00+A11)(B00+B11) -> C00+C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00 (0*2+0), C11 (1*2+1)
    
    # M2: (A10+A11)B00 -> C10-C11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C10 (0*2+1? No k=0, i=1 -> 0*2+1=1? Wait.)
    # W index k*n+i.
    # C10: i=1, k=0 -> 0*2+1 = 1.
    # C11: i=1, k=1 -> 1*2+1 = 3.
    # C00: i=0, k=0 -> 0.
    # C01: i=0, k=1 -> 1.
    # My previous W mapping:
    # W[0] is C00.
    # W[1] is C01? No, k=1, i=0 -> 1*2+0 = 2.
    # Let's re-verify W index.
    # W[k*n+i].
    # k=0, i=0 -> 0. (C00)
    # k=0, i=1 -> 1. (C10)
    # k=1, i=0 -> 2. (C01)
    # k=1, i=1 -> 3. (C11)
    
    # Correcting Strassen W indices:
    # M1: C00 (0), C11 (3).
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: C10 (1), C11 (3).
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11) -> C00+C01
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1 # B01 (0*2+1=1), B11 (1*2+1=3)
    W[0, 2] = 1; W[2, 2] = 1 # C00 (0), C01 (2)
    
    # M4: A11(B10-B00) -> C00+C10
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1 # B10 (1*2+0=2), B00 (0)
    W[0, 3] = 1; W[1, 3] = 1 # C00 (0), C10 (1)
    
    # M5: (A00+A01)B11 -> -C00+C01
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00+C11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Ranks
    # 2x5x5 = 18 (2x2x5) + 18 (2x2x5) + 2 (2x1x5)
    # 1x5x5 = 5
    # Total 43
    
    U_full = np.zeros((n*m, 43))
    V_full = np.zeros((m*p, 43))
    W_full = np.zeros((p*n, 43))
    
    col_offset = 0
    
    # Block 1: 2x2x5 (Rows 0,1 of A; Cols 0,1 of B)
    # 2x2x5 = 2x2x2 (Rank 7) + 2x2x2 (Rank 7) + 2x2x1 (Rank 4)
    # Sub-sub 1: 2x2x2 (B cols 0,1)
    U11, V11, W11 = strassen_222()
    # Embed into full
    # U rows: i in {0,1}, j in {0,1}.
    for i in range(2):
        for j in range(2):
            U_full[i*m + j, col_offset:col_offset+7] = U11[i*2+j]
    # V rows: j in {0,1}, k in {0,1}.
    for j in range(2):
        for k in range(2):
            V_full[j*p + k, col_offset:col_offset+7] = V11[j*2+k]
    # W rows: k in {0,1}, i in {0,1}.
    for k in range(2):
        for i in range(2):
            W_full[k*n + i, col_offset:col_offset+7] = W11[k*2+i]
    col_offset += 7
    
    # Sub-sub 2: 2x2x2 (B cols 2,3)
    U12, V12, W12 = strassen_222()
    for i in range(2):
        for j in range(2):
            U_full[i*m + j, col_offset:col_offset+7] = U12[i*2+j]
    for j in range(2):
        for k in range(2):
            V_full[j*p + (k+2), col_offset:col_offset+7] = V12[j*2+k]
    for k in range(2):
        for i in range(2):
            W_full[(k+2)*n + i, col_offset:col_offset+7] = W12[k*2+i]
    col_offset += 7
    
    # Sub-sub 3: 2x2x1 (B col 4)
    U13, V13, W13 = schoolbook(2, 2, 1)
    for i in range(2):
        for j in range(2):
            U_full[i*m + j, col_offset:col_offset+4] = U13[i*2+j]
    for j in range(2):
        V_full[j*p + 4, col_offset:col_offset+4] = V13[j*1] # V13 shape 2x4. j*1+0
    for k in range(1):
        for i in range(2):
            W_full[(k+4)*n + i, col_offset:col_offset+4] = W13[k*2+i]
    col_offset += 4
    
    # Block 2: 2x1x5 (Rows 0,1 of A; Col 2 of B? No, A is 2x5, B is 5x5.
    # Wait, my split of 2x5x5 was:
    # A1 (2x5) = [A11(2x2) | A12(2x2) | A13(2x1)]
    # B (5x5) = [B11(2x5); B12(2x5); B13(1x5)]
    # So the 3rd part is A13 (2x1) * B13 (1x5).
    # This is 2x1x5.
    U2, V2, W2 = schoolbook(2, 1, 5)
    for i in range(2):
        U_full[i*m + 4, col_offset:col_offset+2] = U2[i*1]
    for j in range(1):
        for k in range(5):
            V_full[(j+4)*p + k, col_offset:col_offset+2] = V2[j*5+k]
    for k in range(5):
        for i in range(2):
            W_full[k*n + i, col_offset:col_offset+2] = W2[k*2+i]
    col_offset += 2
    
    # Block 3: 1x5x5 (Row 2 of A)
    U3, V3, W3 = schoolbook(1, 5, 5)
    for i in range(1):
        for j in range(5):
            U_full[(i+2)*m + j, col_offset:col_offset+5] = U3[i*5+j]
    for j in range(5):
        for k in range(5):
            V_full[j*p + k, col_offset:col_offset+5] = V3[j*5+k]
    for k in range(5):
        for i in range(1):
            W_full[k*n + (i+2), col_offset:col_offset+5] = W3[k*1+i]
    col_offset += 5
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
