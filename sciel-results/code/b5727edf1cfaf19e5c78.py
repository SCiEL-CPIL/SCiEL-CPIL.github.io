import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 5, 5
    # U: (n*m, r) = (10, r)
    # V: (m*p, r) = (25, r)
    # W: (p*n, r) = (10, r)
    
    rank = 41
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients for 2x2 * 2x2
    # U_strassen: 4 entries (x11, x12, x21, x22) -> 7
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # x11
        [0, 0, 0, 0, 1, 0, 1],   # x12
        [0, 1, 0, 0, 0, 1, 0],   # x21
        [1, 1, 0, 1, 0, 0, -1]   # x22
    ], dtype=float)
    
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # y11
        [0, 0, 1, 0, 0, 1, 0],   # y12
        [0, 0, 0, 1, 0, 0, 1],   # y21
        [1, 0, -1, 0, 1, 0, 1]   # y22
    ], dtype=float)
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # c11
        [0, 0, 1, 0, 1, 0, 0],   # c12
        [0, 1, 0, 1, 0, 0, 0],   # c21
        [1, -1, 1, 0, 0, 1, 0]   # c22
    ], dtype=float)
    
    # Schoolbook for 2x2 * 2x1: 4 terms
    # U: x11, x12, x21, x22 each contribute to one term
    # V: y1, y2 each contribute to two terms
    # W: c1, c2 each get two terms
    # Term 1: x11 * y1 -> c1
    # Term 2: x12 * y2 -> c1
    # Term 3: x21 * y1 -> c2
    # Term 4: x22 * y2 -> c2
    U_sb22_21 = np.array([
        [1, 0, 0, 0],  # x11
        [0, 1, 0, 0],  # x12
        [0, 0, 1, 0],  # x21
        [0, 0, 0, 1]   # x22
    ], dtype=float)
    V_sb22_21 = np.array([
        [1, 0, 1, 0],  # y1
        [0, 1, 0, 1]   # y2
    ], dtype=float)
    W_sb22_21 = np.array([
        [1, 1, 0, 0],  # c1
        [0, 0, 1, 1]   # c2
    ], dtype=float)
    
    # Schoolbook for 2x1 * 1x2: 2 terms
    # Term 1: x1 * y1 -> c1
    # Term 2: x2 * y2 -> c2
    U_sb21_12 = np.array([
        [1, 0],  # x1
        [0, 1]   # x2
    ], dtype=float)
    V_sb21_12 = np.array([
        [1, 0],  # y1
        [0, 1]   # y2
    ], dtype=float)
    W_sb21_12 = np.array([
        [1, 0],  # c1
        [0, 1]   # c2
    ], dtype=float)
    
    # Schoolbook for 2x1 * 1x1: 1 term
    U_sb21_11 = np.array([[1], [1]], dtype=float)  # x1, x2
    V_sb21_11 = np.array([[1]], dtype=float)       # y1
    W_sb21_11 = np.array([[1], [1]], dtype=float)  # c1, c2
    
    col = 0
    
    # Block A1 * B_top: A1 is 2x2 (A cols 0,1), B_top is 2x5 (B rows 0,1)
    # B_top split: B11 (cols 0,1), B12 (cols 2,3), B1+ (col 4)
    
    # A1 * B11: Strassen, 2x2 * 2x2
    for i in range(2):
        for j in range(2):
            U[i * m + j, col:col+7] = U_s[i * 2 + j, :]
    for j in range(2):
        for k in range(2):
            V[j * p + k, col:col+7] = V_s[j * 2 + k, :]
    for i in range(2):
        for k in range(2):
            W[k * n + i, col:col+7] = W_s[i * 2 + k, :]
    col += 7
    
    # A1 * B12: Strassen, 2x2 * 2x2
    for i in range(2):
        for j in range(2):
            U[i * m + j, col:col+7] = U_s[i * 2 + j, :]
    for j in range(2):
        for k in range(2, 4):
            V[j * p + k, col:col+7] = V_s[j * 2 + (k-2), :]
    for i in range(2):
        for k in range(2, 4):
            W[k * n + i, col:col+7] = W_s[i * 2 + (k-2), :]
    col += 7
    
    # A1 * B1+: Schoolbook 2x2 * 2x1, 4 terms
    for i in range(2):
        for j in range(2):
            U[i * m + j, col:col+4] = U_sb22_21[i * 2 + j, :]
    for j in range(2):
        V[j * p + 4, col:col+4] = V_sb22_21[j, :]
    for i in range(2):
        W[4 * n + i, col:col+4] = W_sb22_21[i, :]
    col += 4
    
    # Block A2 * B_mid: A2 is 2x2 (A cols 2,3), B_mid is 2x5 (B rows 2,3)
    # B_mid split: B21 (cols 0,1), B22 (cols 2,3), B2+ (col 4)
    
    # A2 * B21: Strassen, 2x2 * 2x2
    for i in range(2):
        for j in range(2, 4):
            U[i * m + j, col:col+7] = U_s[i * 2 + (j-2), :]
    for j in range(2, 4):
        for k in range(2):
            V[j * p + k, col:col+7] = V_s[(j-2) * 2 + k, :]
    for i in range(2):
        for k in range(2):
            W[k * n + i, col:col+7] = W_s[i * 2 + k, :]
    col += 7
    
    # A2 * B22: Strassen, 2x2 * 2x2
    for i in range(2):
        for j in range(2, 4):
            U[i * m + j, col:col+7] = U_s[i * 2 + (j-2), :]
    for j in range(2, 4):
        for k in range(2, 4):
            V[j * p + k, col:col+7] = V_s[(j-2) * 2 + (k-2), :]
    for i in range(2):
        for k in range(2, 4):
            W[k * n + i, col:col+7] = W_s[i * 2 + (k-2), :]
    col += 7
    
    # A2 * B2+: Schoolbook 2x2 * 2x1, 4 terms
    for i in range(2):
        for j in range(2, 4):
            U[i * m + j, col:col+4] = U_sb22_21[i * 2 + (j-2), :]
    for j in range(2, 4):
        V[j * p + 4, col:col+4] = V_sb22_21[j-2, :]
    for i in range(2):
        W[4 * n + i, col:col+4] = W_sb22_21[i, :]
    col += 4
    
    # Block A3 * B_bot: A3 is 2x1 (A col 4), B_bot is 1x5 (B row 4)
    # B_bot split: B_bot0 (cols 0,1), B_bot1 (cols 2,3), B_bot2 (col 4)
    
    # A3 * B_bot0: Schoolbook 2x1 * 1x2, 2 terms
    for i in range(2):
        U[i * m + 4, col:col+2] = U_sb21_12[i, :]
    for k in range(2):
        V[4 * p + k, col:col+2] = V_sb21_12[k, :]
    for i in range(2):
        for k in range(2):
            W[k * n + i, col:col+2] = W_sb21_12[i * 2 + k, :]
    col += 2
    
    # A3 * B_bot1: Schoolbook 2x1 * 1x2, 2 terms
    for i in range(2):
        U[i * m + 4, col:col+2] = U_sb21_12[i, :]
    for k in range(2, 4):
        V[4 * p + k, col:col+2] = V_sb21_12[k-2, :]
    for i in range(2):
        for k in range(2, 4):
            W[k * n + i, col:col+2] = W_sb21_12[i * 2 + (k-2), :]
    col += 2
    
    # A3 * B_bot2: Schoolbook 2x1 * 1x1, 1 term
    for i in range(2):
        U[i * m + 4, col] = U_sb21_11[i, 0]
    V[4 * p + 4, col] = V_sb21_11[0, 0]
    for i in range(2):
        W[4 * n + i, col] = W_sb21_11[i, 0]
    col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    # Verify the decomposition
    # Check: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    print(f"Mismatch at i={i}, j={j}, k={k}: val={val}")
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if not ok:
        # Fallback: schoolbook
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
    
    # Convert to lists
    U_list = [[int(round(x)) for x in row] for row in U]
    V_list = [[int(round(x)) for x in row] for row in V]
    W_list = [[int(round(x)) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
