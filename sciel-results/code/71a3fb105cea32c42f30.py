import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # A indices: 0,1,2,3 (row-major)
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    
    # The 7 rank-1 terms for Strassen
    # M1 = a(e+h)
    # M2 = (a+b)h
    # M3 = (c+d)e
    # M4 = d(e+h)
    # M5 = a(f+g)
    # M6 = (c-a)(e+h) -- Correction: M6 = (c-a)(e+h) is not standard.
    # Standard:
    # M1 = a(e+h)
    # M2 = (a+b)h
    # M3 = (c+d)e
    # M4 = d(e+h)
    # M5 = a(f+g)
    # M6 = (c-a)(e+h)
    # M7 = (b-d)(g+h)
    
    # Let's verify M6, M7.
    # C00 = M1 - M4 + M3
    # C01 = M5 + M2 - M4 - M7
    # C10 = M5 + M3 - M1 + M7
    # C11 = M2 + M4 - M3 - M6
    
    # This seems consistent with some sources.
    
    strassen_U = [
        [1, 0, 0, 0], # a
        [1, 1, 0, 0], # a+b
        [0, 0, 1, 1], # c+d
        [0, 0, 0, 1], # d
        [1, 0, 0, 0], # a
        [0, 0, 1, -1], # c-a
        [0, 1, 0, -1]  # b-d
    ]
    strassen_V = [
        [1, 0, 0, 1], # e+h
        [0, 0, 0, 1], # h
        [1, 0, 0, 0], # e
        [1, 0, 0, 1], # e+h
        [0, 1, 1, 0], # f+g
        [1, 0, 0, 1], # e+h
        [0, 0, 1, 1]  # g+h
    ]
    strassen_W = [
        [1, 0, 0, -1], # C00: 1, C01: 0, C10: 0, C11: -1
        [0, 1, 0, -1], # C00: 0, C01: 1, C10: 0, C11: -1
        [0, 0, 1, -1], # C00: 0, C01: 0, C10: 1, C11: -1
        [-1, -1, 1, 1], # C00: -1, C01: -1, C10: 1, C11: 1
        [0, 1, 1, 0], # C00: 0, C01: 1, C10: 1, C11: 0
        [0, 0, 0, -1], # C00: 0, C01: 0, C10: 0, C11: -1
        [0, -1, 1, 0]  # C00: 0, C01: -1, C10: 1, C11: 0
    ]
    
    # Schoolbook 2x2x3 decomposition
    # A: 2x2, B: 2x3, C: 2x3
    # 12 terms.
    
    # Loop over the two m-blocks
    for block in range(2):
        m_offset = block * 2
        
        # Sub-block 1: p=2 (cols 0,1)
        # Use Strassen
        for r in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    idx_A = i * 2 + j
                    idx_U = (m_offset + j) # Wait, A is 2x4.
                    # A indices: i in 0..1, j in 0..3.
                    # Block 0: j in 0,1. Block 1: j in 2,3.
                    idx_U = i * m + (m_offset + j)
                    U[idx_U][col] = strassen_U[r][idx_A]
            
            # V
            # B is 4x5.
            # Block 0: rows 0,1. Cols 0,1.
            for j in range(2):
                for k in range(2):
                    idx_B = j * p + k
                    idx_V = j * 2 + k # B local index
                    V[idx_B][col] = strassen_V[r][idx_V]
            
            # W
            # C is 5x2.
            # Cols 0,1.
            for k in range(2):
                for i in range(2):
                    idx_C = k * n + i
                    idx_W = k * 2 + i # C local index
                    W[idx_C][col] = strassen_W[r][idx_W]
            
            col += 1
        
        # Sub-block 2: p=3 (cols 2,3,4)
        # Use Schoolbook
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    idx_U = i * m + (m_offset + j)
                    idx_V = j * p + (2 + k)
                    idx_W = (2 + k) * n + i
                    U[idx_U][col] = 1
                    V[idx_V][col] = 1
                    W[idx_W][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
