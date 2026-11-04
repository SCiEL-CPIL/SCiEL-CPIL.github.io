import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the classic Strassen 2x2x2 decomposition with 7 terms.
    U, V, W are 4x7 matrices (row-major indexing).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1: M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 3: M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 4: M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1
    
    # Term 5: M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 6: M6 = (c-d)(f+g)
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (b-d)(e+f)
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[1][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """
    Return schoolbook decomposition for n x m x p multiplication.
    U is (n*m) x (n*m*p), V is (m*p) x (n*m*p), W is (p*n) x (n*m*p).
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _construct_5x5x5():
    n, m, p = 5, 5, 5
    nm, mp, pn = n * m, m * p, p * n
    
    # Blocks:
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # Dimensions of sub-blocks:
    # A11, B11, C11: 2x2
    # A12, B12, C12: 2x3
    # A21, B21, C21: 3x2
    # A22, B22, C22: 3x3
    
    # Ranks:
    # A11 B11: 7 (Strassen)
    # A12 B21: 2*3*2 = 12 (Schoolbook)
    # A11 B12: 2*2*3 = 6 (Schoolbook)
    # A12 B22: 2*3*3 = 18 (Schoolbook)
    # A21 B11: 3*2*2 = 12 (Schoolbook)
    # A22 B21: 3*3*2 = 18 (Schoolbook)
    # A21 B12: 3*2*3 = 18 (Schoolbook)
    # A22 B22: 3*3*3 = 27 (Schoolbook)
    
    total_rank = 7 + 12 + 6 + 18 + 12 + 18 + 18 + 27
    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]
    
    col = 0
    
    # 1. C11 diag: A11 B11 (2x2x2) -> Strassen
    U_s, V_s, W_s = _strassen_2x2()
    for c in range(7):
        for r in range(4):
            # i in 0..1, j in 0..1, k in 0..1
            U[r][col + c] = U_s[r][c]
            V[r][col + c] = V_s[r][c]
            W[r][col + c] = W_s[r][c]
    col += 7
    
    # 2. C11 off: A12 B21 (2x3x2) -> Schoolbook
    # i in 0..1, j in 2..4, k in 0..1
    U_b, V_b, W_b = _schoolbook_block(2, 3, 2)
    for c in range(12):
        for i in range(2):
            for j in range(3):
                # U index: i*2 + (j-2) ? No, U_b is local.
                # Local i: 0..1, Local j: 0..2 (maps to global 2..4)
                # Global U index: i * 5 + (j+2)
                # Global V index: (j+2) * 5 + k
                # Global W index: k * 5 + i
                pass # We will fill manually to avoid index confusion in helper
    
    # Manual filling for clarity and correctness
    
    # 2. C11 off: A12 B21 (i in 0..1, j in 2..4, k in 0..1)
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + i][col] = 1
                col += 1
    
    # 3. C12 left: A11 B12 (i in 0..1, j in 0..1, k in 2..4)
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + i][col] = 1
                col += 1
    
    # 4. C12 right: A12 B22 (i in 0..1, j in 2..4, k in 2..4)
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + i][col] = 1
                col += 1
    
    # 5. C21 top: A21 B11 (i in 2..4, j in 0..1, k in 0..1)
    for i in range(2, 5):
        for j in range(2):
            for k in range(2):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + i][col] = 1
                col += 1
    
    # 6. C21 bottom: A22 B21 (i in 2..4, j in 2..4, k in 0..1)
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + i][col] = 1
                col += 1
    
    # 7. C22 left: A21 B12 (i in 2..4, j in 0..1, k in 2..4)
    for i in range(2, 5):
        for j in range(2):
            for k in range(2, 5):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + i][col] = 1
                col += 1
    
    # 8. C22 diag: A22 B22 (i in 2..4, j in 2..4, k in 2..4)
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}, total_rank

def _verify_fast(n, m, p, U, V, W):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    result, rank = _construct_5x5x5()
    
    if _verify_fast(n, m, p, result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: Pure schoolbook (rank 125)
    U = [[0] * 125 for _ in range(25)]
    V = [[0] * 125 for _ in range(25)]
    W = [[0] * 125 for _ in range(25)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
