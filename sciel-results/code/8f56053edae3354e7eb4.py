"""
Matrix multiplication 2x2 by 2x13.
We decompose the 2x13 matrix into six 2x2 blocks and one 2x1 block.
For each 2x2 block, we use the rank-7 Strassen decomposition.
For the 2x1 block, we use the rank-4 schoolbook decomposition.
Total rank = 6 * 7 + 4 = 46.
"""

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    # U is (n*m, rank) = (4, rank)
    # V is (m*p, rank) = (26, rank)
    # W is (p*n, rank) = (26, rank)
    
    # Strassen 2x2x2 decomposition (rank 7)
    # C = A * B where A is 2x2, B is 2x2
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U indices: i*m + j -> (0,1) for (0,0), (0,1) for (0,1), (1,0) for (1,0), (1,1) for (1,1)
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Strassen U vectors (for A indices 0..3):
    # M1: A00+A11 -> [1, 0, 0, 1]
    # M2: A10+A11 -> [0, 0, 1, 1]
    # M3: A00       -> [1, 0, 0, 0]
    # M4: A11       -> [0, 0, 0, 1]
    # M5: A00+A01   -> [1, 1, 0, 0]
    # M6: A10-A00   -> [-1, 0, 1, 0]
    # M7: A01-A11   -> [0, 1, 0, -1]
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # Strassen V vectors (for B indices 0..3 in local 2x2):
    # M1: B00+B11 -> [1, 0, 0, 1]
    # M2: B00     -> [1, 0, 0, 0]
    # M3: B10-B11 -> [0, 0, 1, -1]
    # M4: B01-B00 -> [-1, 1, 0, 0]
    # M5: B11     -> [0, 0, 0, 1]
    # M6: B00+B01 -> [1, 1, 0, 0]
    # M7: B10+B11 -> [0, 0, 1, 1]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Strassen W vectors (for C indices 0..3 in local 2x2):
    # M1: C00+C11 -> [1, 0, 0, 1]
    # M2: -C10-C11 -> [0, 0, -1, -1]
    # M3: C01+C10 -> [0, 1, 1, 0]
    # M4: C00+C10 -> [1, 0, 1, 0]
    # M5: -C00+C01 -> [-1, 1, 0, 0]
    # M6: C11      -> [0, 0, 0, 1]
    # M7: C00-C01 -> [1, -1, 0, 0]
    strassen_w = [
        [1, 0, 0, 1],
        [0, 0, -1, -1],
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, -1, 0, 0]
    ]
    
    # Schoolbook 2x2x1 decomposition (rank 4)
    # C = A * B where A is 2x2, B is 2x1
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    #
    # Terms:
    # T1: A00 * B0 -> C0
    # T2: A01 * B1 -> C0
    # T3: A10 * B0 -> C1
    # T4: A11 * B1 -> C1
    schoolbook_u = [
        [1, 0, 0, 0],  # A00
        [0, 1, 0, 0],  # A01
        [0, 0, 1, 0],  # A10
        [0, 0, 0, 1]   # A11
    ]
    schoolbook_v = [
        [1, 0],  # B0
        [0, 1],  # B1
        [1, 0],  # B0
        [0, 1]   # B1
    ]
    schoolbook_w = [
        [1, 0],  # C0
        [1, 0],  # C0
        [0, 1],  # C1
        [0, 1]   # C1
    ]
    
    total_rank = 6 * 7 + 4  # 42 + 4 = 46
    U = [[0] * total_rank for _ in range(n * m)]  # (4, 46)
    V = [[0] * total_rank for _ in range(m * p)]  # (26, 46)
    W = [[0] * total_rank for _ in range(p * n)]  # (26, 46)
    
    col = 0
    # Process 6 blocks of size 2x2 (columns 0-11)
    for block in range(6):
        # Local B indices: (0,0) -> global (0, 2*block), (0,1) -> (0, 2*block+1),
        #                  (1,0) -> (1, 2*block), (1,1) -> (1, 2*block+1)
        # Local C indices: (0,0) -> global (2*block, 0), (0,1) -> (2*block, 1),
        #                  (1,0) -> (2*block+1, 0), (1,1) -> (2*block+1, 1)
        for t in range(7):
            # U part
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] += strassen_u[t][i * 2 + j]
            
            # V part
            for j in range(2):
                for k in range(2):
                    # global k = 2*block + k
                    V[j * p + (2 * block + k)][col] += strassen_v[t][j * 2 + k]
            
            # W part
            for k in range(2):
                for i in range(2):
                    # global k = 2*block + k, global i = i
                    W[(2 * block + k) * n + i][col] += strassen_w[t][k * 2 + i]
            
            col += 1
    
    # Process last block of size 2x1 (column 12)
    for t in range(4):
        # U part
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] += schoolbook_u[t][i * 2 + j]
        
        # V part
        for j in range(2):
            # global k = 12
            V[j * p + 12][col] += schoolbook_v[t][j]
        
        # W part
        for i in range(2):
            # global k = 12, global i = i
            W[12 * n + i][col] += schoolbook_w[t][i]
        
        col += 1
    
    return {"U": U, "V": V, "W": W}
