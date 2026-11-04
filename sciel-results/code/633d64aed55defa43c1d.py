import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 34
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a term: U[ua, col] += cu, V[va, col] += cv, W[wa, col] += cw
    def add_term(ua, va, wa, cu, cv, cw):
        nonlocal col
        U[ua, col] += cu
        V[va, col] += cv
        W[wa, col] += cw

    # Indices:
    # A: i*m + j
    # B: j*p + k
    # C: k*n + i
    
    # Part 1: A_TL (2x2) * B_top_L (2x2) -> C_top_L (2x2)
    # Strassen 2x2
    # m1 = (A00+A11)(B00+B11) -> C00, C11
    U[0, col] += 1; U[4, col] += 1
    V[0, col] += 1; V[5, col] += 1
    W[0, col] += 1; W[4, col] += 1
    col += 1
    
    # m2 = (A10+A11)B00 -> C10, -C11
    U[3, col] += 1; U[4, col] += 1
    V[0, col] += 1
    W[1, col] += 1; W[4, col] += -1
    col += 1
    
    # m3 = A00(B01-B11) -> C01, C11
    U[0, col] += 1
    V[1, col] += 1; V[5, col] += -1
    W[3, col] += 1; W[4, col] += 1
    col += 1
    
    # m4 = A11(B10-B00) -> C00, C10
    U[4, col] += 1
    V[4, col] += 1; V[0, col] += -1
    W[0, col] += 1; W[1, col] += 1
    col += 1
    
    # m5 = (A00+A01)B11 -> C01, -C00
    U[0, col] += 1; U[1, col] += 1
    V[5, col] += 1
    W[3, col] += 1; W[0, col] += -1
    col += 1
    
    # m6 = (A10-A00)(B00+B01) -> C11
    U[3, col] += 1; U[0, col] += -1
    V[0, col] += 1; V[1, col] += 1
    W[4, col] += 1
    col += 1
    
    # m7 = (A01-A11)(B10+B11) -> C00
    U[1, col] += 1; U[4, col] += -1
    V[4, col] += 1; V[5, col] += 1
    W[0, col] += 1
    col += 1
    
    # Part 2: A_TL (2x2) * B_top_R (2x2) -> C_top_R (2x2)
    # B cols 2,3. Indices: B02=2, B03=3, B12=6, B13=7
    # C cols 2,3. Indices: C02=6, C03=7, C12=9, C13=10
    
    # m1' = (A00+A11)(B02+B12) -> C02, C12
    U[0, col] += 1; U[4, col] += 1
    V[2, col] += 1; V[6, col] += 1
    W[6, col] += 1; W[9, col] += 1
    col += 1
    
    # m2' = (A10+A11)B02 -> C12, -C12? No, C12 is k=2, i=1.
    # Wait, C indices: k*n+i.
    # C12: k=2, i=1 -> 2*3+1 = 7? No.
    # C is 3x4. k is col, i is row.
    # C[i,k] = sum A[i,j]B[j,k].
    # W index is k*n + i.
    # C12 means i=1, k=2. Index = 2*3 + 1 = 7.
    # C13 means i=1, k=3. Index = 3*3 + 1 = 10.
    # C02 means i=0, k=2. Index = 2*3 + 0 = 6.
    # C03 means i=0, k=3. Index = 3*3 + 0 = 9.
    
    # Let's re-verify C indices.
    # C00: i=0,k=0 -> 0
    # C01: i=0,k=1 -> 3
    # C02: i=0,k=2 -> 6
    # C03: i=0,k=3 -> 9
    # C10: i=1,k=0 -> 1
    # C11: i=1,k=1 -> 4
    # C12: i=1,k=2 -> 7
    # C13: i=1,k=3 -> 10
    # C20: i=2,k=0 -> 2
    # C21: i=2,k=1 -> 5
    # C22: i=2,k=2 -> 8
    # C23: i=2,k=3 -> 11
    
    # My previous C indices for Part 1 were:
    # C00=0, C01=3, C10=1, C11=4. Correct.
    
    # Part 2 C indices:
    # C02=6, C03=9, C12=7, C13=10.
    
    # m2' = (A10+A11)B02 -> C12 (coeff 1), -C12? No.
    # Strassen m2 contributes to C10 and -C11 in the 2x2 block.
    # Here the 2x2 block is C_top_R.
    # Local C10 -> Global C12 (7).
    # Local C11 -> Global C13 (10).
    
    U[3, col] += 1; U[4, col] += 1
    V[2, col] += 1
    W[7, col] += 1; W[10, col] += -1
    col += 1
    
    # m3' = A00(B03-B12) -> C03 (3), C13 (1)
    # Local C01 -> Global C03 (9).
    # Local C11 -> Global C13 (10).
    U[0, col] += 1
    V[3, col] += 1; V[6, col] += -1
    W[9, col] += 1; W[10, col] += 1
    col += 1
    
    # m4' = A11(B12-B02) -> C02 (1), C12 (1)
    # Local C00 -> Global C02 (6).
    # Local C10 -> Global C12 (7).
    U[4, col] += 1
    V[6, col] += 1; V[2, col] += -1
    W[6, col] += 1; W[7, col] += 1
    col += 1
    
    # m5' = (A00+A01)B12 -> C03 (1), -C02 (-1)
    # Local C01 -> Global C03 (9).
    # Local C00 -> Global C02 (6).
    U[0, col] += 1; U[1, col] += 1
    V[6, col] += 1
    W[9, col] += 1; W[6, col] += -1
    col += 1
    
    # m6' = (A10-A00)(B02+B03) -> C13 (1)
    # Local C11 -> Global C13 (10).
    U[3, col] += 1; U[0, col] += -1
    V[2, col] += 1; V[3, col] += 1
    W[10, col] += 1
    col += 1
    
    # m7' = (A01-A11)(B12+B13) -> C02 (1)
    # Local C00 -> Global C02 (6).
    U[1, col] += 1; U[4, col] += -1
    V[6, col] += 1; V[7, col] += 1
    W[6, col] += 1
    col += 1
    
    # Part 3: A_TR (2x1) * B_bot (1x4)
    # A02=2, A12=5
    # B20=8, B21=9, B22=10, B23=11
    # C00=0, C01=3, C02=6, C03=9
    # C10=1, C11=4, C12=7, C13=10
    
    U[2, col] += 1; V[8, col] += 1; W[0, col] += 1; col += 1
    U[2, col] += 1; V[9, col] += 1; W[3, col] += 1; col += 1
    U[2, col] += 1; V[10, col] += 1; W[6, col] += 1; col += 1
    U[2, col] += 1; V[11, col] += 1; W[9, col] += 1; col += 1
    
    U[5, col] += 1; V[8, col] += 1; W[1, col] += 1; col += 1
    U[5, col] += 1; V[9, col] += 1; W[4, col] += 1; col += 1
    U[5, col] += 1; V[10, col] += 1; W[7, col] += 1; col += 1
    U[5, col] += 1; V[11, col] += 1; W[10, col] += 1; col += 1
    
    # Part 4: A_BL (1x2) * B_top (2x4)
    # A20=6, A21=7
    # B00=0, B01=1, B02=2, B03=3
    # B10=4, B11=5, B12=6, B13=7
    # C20=2, C21=5, C22=8, C23=11
    
    # C20 = A20*B00 + A21*B10
    U[6, col] += 1; V[0, col] += 1; W[2, col] += 1; col += 1
    U[7, col] += 1; V[4, col] += 1; W[2, col] += 1; col += 1
    
    # C21 = A20*B01 + A21*B11
    U[6, col] += 1; V[1, col] += 1; W[5, col] += 1; col += 1
    U[7, col] += 1; V[5, col] += 1; W[5, col] += 1; col += 1
    
    # C22 = A20*B02 + A21*B12
    U[6, col] += 1; V[2, col] += 1; W[8, col] += 1; col += 1
    U[7, col] += 1; V[6, col] += 1; W[8, col] += 1; col += 1
    
    # C23 = A20*B03 + A21*B13
    U[6, col] += 1; V[3, col] += 1; W[11, col] += 1; col += 1
    U[7, col] += 1; V[7, col] += 1; W[11, col] += 1; col += 1
    
    # Part 5: A_BR (1x1) * B_bot (1x4)
    # A22=8
    # B20=8, B21=9, B22=10, B23=11
    # C20=2, C21=5, C22=8, C23=11
    
    U[8, col] += 1; V[8, col] += 1; W[2, col] += 1; col += 1
    U[8, col] += 1; V[9, col] += 1; W[5, col] += 1; col += 1
    U[8, col] += 1; V[10, col] += 1; W[8, col] += 1; col += 1
    U[8, col] += 1; V[11, col] += 1; W[11, col] += 1; col += 1
    
    # Verify
    T_approx = np.einsum('ac,bc,dc->abd', U, V, W)
    
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                valid_indices.append((alpha, beta, gamma))
                
    total_indices = []
    for alpha in range(n * m):
        for beta in range(m * p):
            for gamma in range(p * n):
                total_indices.append((alpha, beta, gamma))
                
    valid_set = set(valid_indices)
    invalid_indices = [idx for idx in total_indices if idx not in valid_set]
    
    for (a, b, g) in valid_indices:
        if not np.isclose(T_approx[a, b, g], 1.0, atol=1e-9):
            # Fallback to schoolbook
            rank_fb = n * m * p
            U_fb = [[0] * rank_fb for _ in range(n * m)]
            V_fb = [[0] * rank_fb for _ in range(m * p)]
            W_fb = [[0] * rank_fb for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][column] = 1
                        V_fb[j * p + k][column] = 1
                        W_fb[k * n + i][column] = 1
                        column += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
            
    for (a, b, g) in invalid_indices:
        if not np.isclose(T_approx[a, b, g], 0.0, atol=1e-9):
            rank_fb = n * m * p
            U_fb = [[0] * rank_fb for _ in range(n * m)]
            V_fb = [[0] * rank_fb for _ in range(m * p)]
            W_fb = [[0] * rank_fb for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][column] = 1
                        V_fb[j * p + k][column] = 1
                        W_fb[k * n + i][column] = 1
                        column += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
