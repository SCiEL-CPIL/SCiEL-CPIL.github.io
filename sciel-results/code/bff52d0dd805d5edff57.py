import numpy as np
import time

DIMENSIONS = [2, 3, 3]
_START_TIME = time.time()
TIME_LIMIT = 120  # seconds, with safety margin

def _time_remaining():
    return TIME_LIMIT - 10 - (time.time() - _START_TIME)  # 10s safety margin

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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

def _build_strassen_2x2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) as lists.
    
    Strassen's 7 products:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # U: A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V: B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # W: C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1: (A00+A11)(B00+B11) -> contributes to C00, C11
    U[0][0] += 1  # A00
    U[3][0] += 1  # A11
    V[0][0] += 1  # B00
    V[3][0] += 1  # B11
    W[0][0] += 1  # C00
    W[3][0] += 1  # C11
    
    # M2: (A10+A11)B00 -> contributes to C10, C11 (negative)
    U[2][1] += 1  # A10
    U[3][1] += 1  # A11
    V[0][1] += 1  # B00
    W[2][1] += 1  # C10
    W[3][1] -= 1  # C11
    
    # M3: A00(B01-B11) -> contributes to C01, C11
    U[0][2] += 1  # A00
    V[1][2] += 1  # B01
    V[3][2] -= 1  # B11
    W[1][2] += 1  # C01
    W[3][2] += 1  # C11
    
    # M4: A11(B10-B00) -> contributes to C00, C10
    U[3][3] += 1  # A11
    V[2][3] += 1  # B10
    V[0][3] -= 1  # B00
    W[0][3] += 1  # C00
    W[2][3] += 1  # C10
    
    # M5: (A00+A01)B11 -> contributes to C00 (negative), C01
    U[0][4] += 1  # A00
    U[1][4] += 1  # A01
    V[3][4] += 1  # B11
    W[0][4] -= 1  # C00
    W[1][4] += 1  # C01
    
    # M6: (A10-A00)(B00+B01) -> contributes to C11
    U[2][5] += 1  # A10
    U[0][5] -= 1  # A00
    V[0][5] += 1  # B00
    V[1][5] += 1  # B01
    W[3][5] += 1  # C11
    
    # M7: (A01-A11)(B10+B11) -> contributes to C00
    U[1][6] += 1  # A01
    U[3][6] -= 1  # A11
    V[2][6] += 1  # B10
    V[3][6] += 1  # B11
    W[0][6] += 1  # C00
    
    return U, V, W

def _build_decomposed_2x3x3():
    """
    Decompose 2x3x3 into:
    - Part 1: 2x2x2 (columns j=0,1; rows k=0,1) -> Strassen rank 7
    - Part 2: 2x2x1 (columns j=0,1; row k=2) -> schoolbook rank 4
    - Part 3: 2x1x3 (column j=2; rows k=0,1,2) -> schoolbook rank 6
    Total rank: 17
    """
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]  # 6x17
    V = [[0] * rank for _ in range(m * p)]  # 9x17
    W = [[0] * rank for _ in range(p * n)]  # 6x17
    
    # --- Part 1: 2x2x2 block (j in {0,1}, k in {0,1}) ---
    # A[i, j] with i in {0,1}, j in {0,1}: index = i*3 + j
    # B[j, k] with j in {0,1}, k in {0,1}: index = j*3 + k
    # C[k, i] with k in {0,1}, i in {0,1}: index = k*2 + i
    U1, V1, W1 = _build_strassen_2x2x2()
    for r in range(7):
        for a_idx in range(4):
            i_local = a_idx // 2
            j_local = a_idx % 2
            a_global = i_local * 3 + j_local
            U[a_global][r] = U1[a_idx][r]
        
        for b_idx in range(4):
            j_local = b_idx // 2
            k_local = b_idx % 2
            b_global = j_local * 3 + k_local
            V[b_global][r] = V1[b_idx][r]
        
        for c_idx in range(4):
            k_local = c_idx // 2
            i_local = c_idx % 2
            c_global = k_local * 2 + i_local
            W[c_global][r] = W1[c_idx][r]
    
    # --- Part 2: 2x2x1 block (j in {0,1}, k=2) ---
    # A[i, j] with i in {0,1}, j in {0,1}: index = i*3 + j
    # B[j, 2] with j in {0,1}: index = j*3 + 2
    # C[2, i] with i in {0,1}: index = 2*2 + i = 4 + i
    # Schoolbook: 4 terms
    col_offset = 7
    for i in range(2):
        for j in range(2):
            col = col_offset + i * 2 + j
            a_global = i * 3 + j
            b_global = j * 3 + 2
            c_global = 4 + i
            U[a_global][col] = 1
            V[b_global][col] = 1
            W[c_global][col] = 1
    
    # --- Part 3: 2x1x3 block (j=2, k in {0,1,2}) ---
    # A[i, 2] with i in {0,1}: index = i*3 + 2
    # B[2, k] with k in {0,1,2}: index = 2*3 + k = 6 + k
    # C[k, i] with k in {0,1,2}, i in {0,1}: index = k*2 + i
    # Schoolbook: 6 terms
    col_offset = 11
    for i in range(2):
        for k in range(3):
            col = col_offset + i * 3 + k
            a_global = i * 3 + 2
            b_global = 6 + k
            c_global = k * 2 + i
            U[a_global][col] = 1
            V[b_global][col] = 1
            W[c_global][col] = 1
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification of the decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(val - 1) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Try the decomposed approach (rank 17)
    try:
        U, V, W = _build_decomposed_2x3x3()
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
