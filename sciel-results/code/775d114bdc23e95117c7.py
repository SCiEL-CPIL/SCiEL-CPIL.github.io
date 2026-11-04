import time

DIMENSIONS = [2, 4, 4]

def _strassen_2x2():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    
    For C = A*B where A,B,C are 2x2:
    A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    
    Strassen's 7 products:
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)(b11)
    M3 = (a11)(b12-b22)
    M4 = (a22)(b21-b11)
    M5 = (a11+a12)(b22)
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    C reconstruction:
    c11 = M1 + M4 - M5 + M7
    c12 = M3 + M5
    c21 = M2 + M4
    c22 = M1 - M2 + M3 + M6
    """
    # A entries: a11=0, a12=1, a21=2, a22=3
    # B entries: b11=0, b12=1, b21=2, b22=3
    # C entries: c11=0, c12=1, c21=2, c22=3
    
    return [
        # M1 = (a11+a22)(b11+b22)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (a21+a22)(b11)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = (a11)(b12-b22)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = (a22)(b21-b11)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (a11+a12)(b22)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (a21-a11)(b11+b12)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (a12-a22)(b21+b22)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Uses Strassen's 7-multiplication algorithm on each of the four 2x2 block products:
      A1*B11, A2*B21, A1*B12, A2*B22
    
    Total rank: 4 * 7 = 28.
    
    Interface:
      U shape: (n*m, rank) = (8, 28)
      V shape: (m*p, rank) = (16, 28)  
      W shape: (p*n, rank) = (8, 28)
    
    Indexing:
      A[i,j]: i in {0,1}, j in {0,1,2,3}. U index: i*m + j
      B[j,k]: j in {0,1,2,3}, k in {0,1,2,3}. V index: j*p + k
      C[i,k]: i in {0,1}, k in {0,1,2,3}. W index: k*n + i (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    strassen_patterns = _strassen_2x2()
    
    # Four block products:
    # Block 1: A1 * B11 -> C[:, 0:2]
    #   A1 = A[:, 0:2], B11 = B[0:2, 0:2]
    # Block 2: A2 * B21 -> C[:, 0:2]
    #   A2 = A[:, 2:4], B21 = B[2:4, 0:2]
    # Block 3: A1 * B12 -> C[:, 2:4]
    #   A1 = A[:, 0:2], B12 = B[0:2, 2:4]
    # Block 4: A2 * B22 -> C[:, 2:4]
    #   A2 = A[:, 2:4], B22 = B[2:4, 2:4]
    
    # Block definitions: (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    blocks = [
        (0, 0, 0, 0),  # A1*B11
        (2, 2, 0, 0),  # A2*B21
        (0, 0, 2, 2),  # A1*B12
        (2, 2, 2, 2),  # A2*B22
    ]
    
    col_idx = 0
    for a_off, b_row_off, b_col_off, c_off in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
            # Fill U: A entries
            # local_idx: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            # global: i = local_idx // 2, j = a_off + local_idx % 2
            for local_idx, val in u_coeffs.items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                g_idx = local_r * m + (a_off + local_c)
                U[g_idx][col_idx] = val
            
            # Fill V: B entries
            # local_idx: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            # global: j = b_row_off + local_idx // 2, k = b_col_off + local_idx % 2
            for local_idx, val in v_coeffs.items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                g_idx = (b_row_off + local_r) * p + (b_col_off + local_c)
                V[g_idx][col_idx] = val
            
            # Fill W: C entries (trace convention: W[k*n + i])
            # local_idx: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            # global: i = local_idx // 2, k = c_off + local_idx % 2
            # W index: k * n + i
            for local_idx, val in w_coeffs.items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                g_i = local_r
                g_k = c_off + local_c
                g_idx = g_k * n + g_i
                W[g_idx][col_idx] = val
            
            col_idx += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
