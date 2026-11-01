import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def check_decomposition(U, V, W, n, m, p):
    """Verify the tensor decomposition identity."""
    r = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_arr = np.array(U, dtype=float) if isinstance(U, list) else U
    V_arr = np.array(V, dtype=float) if isinstance(V, list) else V
    W_arr = np.array(W, dtype=float) if isinstance(W, list) else W
    
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            s = np.sum(U_arr[a * m + b] * V_arr[c * p + d] * W_arr[k * n + i])
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 2, 3, 6
    
    # Build 2x3x3 decomposition (rank 17)
    # Split A (2x3) into A_left (2x2) and A_col2 (2x1)
    # Split B (3x3) into B_top (2x2), B_col2 (2x1), B_bottom (1x3)
    # C (2x3) has columns C_col0, C_col1, C_col2
    
    U233 = np.zeros((6, 17))
    V233 = np.zeros((9, 17))
    W233 = np.zeros((6, 17))
    
    # Part 1: M(2,2,2) - Strassen for A_left * B_top -> C_top
    # A indices: i*3+j. A_left: j=0,1. So A00=0, A01=1, A10=3, A11=4
    # B indices: j*3+k. B_top: j=0,1, k=0,1. So B00=0, B01=1, B10=3, B11=4
    # C indices: k*2+i. C_top: k=0,1. So C00=0, C10=1, C01=2, C11=3
    
    # P1: (A00+A11)(B00+B11) -> C00, C11
    U233[0, 0] = 1; U233[4, 0] = 1
    V233[0, 0] = 1; V233[4, 0] = 1
    W233[0, 0] = 1; W233[3, 0] = 1
    
    # P2: (A10+A11)B00 -> C10
    U233[3, 1] = 1; U233[4, 1] = 1
    V233[0, 1] = 1
    W233[1, 1] = 1
    
    # P3: A00(B01-B11) -> C01, C11
    U233[0, 2] = 1
    V233[1, 2] = 1; V233[4, 2] = -1
    W233[2, 2] = 1; W233[3, 2] = 1
    
    # P4: A11(B10-B00) -> C00, C10
    U233[4, 3] = 1
    V233[3, 3] = 1; V233[0, 3] = -1
    W233[0, 3] = 1; W233[1, 3] = 1
    
    # P5: (A00+A01)B11 -> C00, C01
    U233[0, 4] = 1; U233[1, 4] = 1
    V233[4, 4] = 1
    W233[0, 4] = -1; W233[2, 4] = 1
    
    # P6: (A10-A00)(B00+B01) -> C11
    U233[3, 5] = 1; U233[0, 5] = -1
    V233[0, 5] = 1; V233[1, 5] = 1
    W233[3, 5] = 1
    
    # P7: (A01-A11)(B10+B11) -> C00, C11
    U233[1, 6] = 1; U233[4, 6] = -1
    V233[3, 6] = 1; V233[4, 6] = 1
    W233[0, 6] = 1; W233[3, 6] = 1
    
    # Part 2: M(2,2,1) - A_left * B_col2 -> C_col2
    # B_col2: j=0,1, k=2. So B02=2, B12=5
    # C_col2: k=2. C02=4, C12=5
    
    # Q1: A00 * B02 -> C02
    U233[0, 7] = 1
    V233[2, 7] = 1
    W233[4, 7] = 1
    
    # Q2: A01 * B12 -> C02
    U233[1, 8] = 1
    V233[5, 8] = 1
    W233[4, 8] = 1
    
    # Q3: A10 * B02 -> C12
    U233[3, 9] = 1
    V233[2, 9] = 1
    W233[5, 9] = 1
    
    # Q4: A11 * B12 -> C12
    U233[4, 10] = 1
    V233[5, 10] = 1
    W233[5, 10] = 1
    
    # Part 3: M(2,1,3) - A_col2 * B_bottom -> C_bottom_cols
    # A_col2: i=0,1, j=2. So A02=2, A12=5
    # B_bottom: j=2, k=0,1,2. So B20=6, B21=7, B22=8
    # C: all columns. C00=0, C10=1, C01=2, C11=3, C02=4, C12=5
    
    # R1: A02 * B20 -> C00
    U233[2, 11] = 1
    V233[6, 11] = 1
    W233[0, 11] = 1
    
    # R2: A02 * B21 -> C01
    U233[2, 12] = 1
    V233[7, 12] = 1
    W233[2, 12] = 1
    
    # R3: A02 * B22 -> C02
    U233[2, 13] = 1
    V233[8, 13] = 1
    W233[4, 13] = 1
    
    # R4: A12 * B20 -> C10
    U233[5, 14] = 1
    V233[6, 14] = 1
    W233[1, 14] = 1
    
    # R5: A12 * B21 -> C11
    U233[5, 15] = 1
    V233[7, 15] = 1
    W233[3, 15] = 1
    
    # R6: A12 * B22 -> C12
    U233[5, 16] = 1
    V233[8, 16] = 1
    W233[5, 16] = 1
    
    # Verify 2x3x3 decomposition
    if not check_decomposition(U233, V233, W233, 2, 3, 3):
        # Fallback to schoolbook
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Combine two 2x3x3 decompositions to form 2x3x6
    # B (3x6) = [B1 | B2] where B1, B2 are 3x3
    # C (2x6) = [C1 | C2] where C1, C2 are 2x3
    
    r233 = 17
    U_comb = np.zeros((6, 2 * r233))
    U_comb[:, :r233] = U233
    U_comb[:, r233:] = U233
    
    V_comb = np.zeros((18, 2 * r233))
    # V index: j*6+k
    # For B1 (k=0,1,2): V[j*6+k] uses V233[j*3+k]
    # For B2 (k=3,4,5): V[j*6+k] uses V233[j*3+(k-3)]
    for j in range(3):
        for k in range(3):
            V_comb[j * 6 + k, :r233] = V233[j * 3 + k]
            V_comb[j * 6 + 3 + k, r233:] = V233[j * 3 + k]
    
    W_comb = np.zeros((12, 2 * r233))
    # W index: k*2+i
    # For C1 (k=0,1,2): W[k*2+i] uses W233[k*2+i]
    # For C2 (k=3,4,5): W[k*2+i] uses W233[(k-3)*2+i]
    for k in range(3):
        for i in range(2):
            W_comb[k * 2 + i, :r233] = W233[k * 2 + i]
            W_comb[(k + 3) * 2 + i, r233:] = W233[k * 2 + i]
    
    # Verify final decomposition
    if check_decomposition(U_comb, V_comb, W_comb, 2, 3, 6):
        return {"U": U_comb.tolist(), "V": V_comb.tolist(), "W": W_comb.tolist()}
    
    # Fallback to schoolbook
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
