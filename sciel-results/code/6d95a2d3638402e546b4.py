import time

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5 second safety margin
    
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to map matrix indices to flat indices
    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return i * n + j # W[k*n+i] where k is row, i is col of C. C[k,i] = sum A[i,j]B[j,k]
    
    # Strassen's 7 multiplications
    # M1 = (A00 + A11) * (B00 + B11)
    U[u_idx(0, 0)][0] = 1; U[u_idx(1, 1)][0] = 1
    V[v_idx(0, 0)][0] = 1; V[v_idx(1, 1)][0] = 1
    # C00 += M1, C11 += M1
    W[w_idx(0, 0)][0] = 1; W[w_idx(1, 1)][0] = 1
    
    # M2 = (A10 + A11) * B00
    U[u_idx(1, 0)][1] = 1; U[u_idx(1, 1)][1] = 1
    V[v_idx(0, 0)][1] = 1
    # C10 += M2, C11 -= M2
    W[w_idx(1, 0)][1] = 1; W[w_idx(1, 1)][1] = -1
    
    # M3 = A00 * (B01 - B11)
    U[u_idx(0, 0)][2] = 1
    V[v_idx(0, 1)][2] = 1; V[v_idx(1, 1)][2] = -1
    # C01 += M3, C11 += M3
    W[w_idx(0, 1)][2] = 1; W[w_idx(1, 1)][2] = 1
    
    # M4 = A11 * (B10 - B00)
    U[u_idx(1, 1)][3] = 1
    V[v_idx(1, 0)][3] = 1; V[v_idx(0, 0)][3] = -1
    # C00 += M4, C10 += M4
    W[w_idx(0, 0)][3] = 1; W[w_idx(1, 0)][3] = 1
    
    # M5 = (A00 + A01) * B11
    U[u_idx(0, 0)][4] = 1; U[u_idx(0, 1)][4] = 1
    V[v_idx(1, 1)][4] = 1
    # C00 -= M5, C01 += M5
    W[w_idx(0, 0)][4] = -1; W[w_idx(0, 1)][4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[u_idx(1, 0)][5] = 1; U[u_idx(0, 0)][5] = -1
    V[v_idx(0, 0)][5] = 1; V[v_idx(0, 1)][5] = 1
    # C11 += M6
    W[w_idx(1, 1)][5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[u_idx(0, 1)][6] = 1; U[u_idx(1, 1)][6] = -1
    V[v_idx(1, 0)][6] = 1; V[v_idx(1, 1)][6] = 1
    # C00 += M7
    W[w_idx(0, 0)][6] = 1

    # Check deadline
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
