import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen blocks
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B00)
    # P4 = A11(B01-B11)
    # P5 = (A00+A10)(B00+B01)
    # P6 = (A01-A11)(B10+B11)
    # P7 = (A01-A00)(B00+B10)
    
    # M00 = P1 - P2 + P4
    # M01 = P3 + P5
    # M10 = P6 + P7
    # M11 = P1 - P3 + P2
    
    strassen_U = [
        [[1, 0], [0, 1]], # A00+A11
        [[0, 0], [1, 1]], # A10+A11
        [[1, 0], [0, 0]], # A00
        [[0, 0], [0, 1]], # A11
        [[1, 1], [0, 0]], # A00+A10
        [[0, 1], [0, -1]], # A01-A11
        [[0, 1], [-1, 0]], # A01-A00
    ]
    strassen_V = [
        [[1, 1], [1, 1]], # B00+B11 (Wait, B is 2x2. B00, B01, B10, B11)
        # B00 is top-left. B01 is top-right. B10 is bottom-left. B11 is bottom-right.
        # Indices: B[j,k].
        # B00: j=0,k=0. B01: j=0,k=1. B10: j=1,k=0. B11: j=1,k=1.
        [[1, 0], [1, 1]], # B00+B11? No.
        # P1: (A00+A11)(B00+B11). B part is B00+B11.
        # B00 is at (0,0), B11 is at (1,1).
        # So V matrix is [[1, 0], [0, 1]]? No, B00+B11 means element (0,0) is 1, (1,1) is 1.
        # So [[1, 0], [0, 1]].
        [[1, 0], [0, 1]], # B00+B11
        [[1, 0], [0, 0]], # B00
        [[0, 0], [1, 0]], # B10? No, P3 = A00(B10-B00). B part is B10-B00.
        # B10 is (1,0), B00 is (0,0).
        [[-1, 0], [1, 0]], # B10-B00
        [[0, 1], [0, -1]], # B01-B11. B01 is (0,1), B11 is (1,1).
        [[1, 1], [0, 0]], # B00+B01. B00 (0,0), B01 (0,1).
        [[0, 0], [1, 1]], # B10+B11. B10 (1,0), B11 (1,1).
        [[1, 1], [0, 0]], # B00+B10? No, P7 = (A01-A00)(B00+B10).
        # B00 (0,0), B10 (1,0).
        [[1, 0], [1, 0]], # B00+B10
    ]
    strassen_W = [
        [[1, 0], [0, 1]], # P1 coeff for M00, M11
        [[-1, 0], [0, 1]], # P2 coeff for M00, M11
        [[0, 1], [1, 0]], # P3 coeff for M01, M11
        [[1, 0], [0, 0]], # P4 coeff for M00
        [[0, 1], [0, 0]], # P5 coeff for M01
        [[0, 0], [1, 0]], # P6 coeff for M10
        [[0, 0], [1, 0]], # P7 coeff for M10
    ]
    # Check W:
    # M00 = P1 - P2 + P4 -> W00: P1=1, P2=-1, P4=1.
    # M01 = P3 + P5 -> W01: P3=1, P5=1.
    # M10 = P6 + P7 -> W10: P6=1, P7=1.
    # M11 = P1 - P3 + P2 -> W11: P1=1, P2=1, P3=-1.
    
    # Correct W:
    strassen_W = [
        [[1, 0], [0, 1]], # P1
        [[-1, 0], [0, 1]], # P2
        [[0, 1], [-1, 0]], # P3
        [[1, 0], [0, 0]], # P4
        [[0, 1], [0, 0]], # P5
        [[0, 0], [1, 0]], # P6
        [[0, 0], [1, 0]], # P7
    ]
    
    # Blocks:
    # 0: A0 B00 -> C0
    # 1: A1 B10 -> C0
    # 2: A0 B01 -> C1
    # 3: A1 B11 -> C1
    
    col = 0
    for block in range(4):
        # Determine A block (0 or 1) and B block (0,1,2,3)
        # Block 0: A0, B00
        # Block 1: A1, B10
        # Block 2: A0, B01
        # Block 3: A1, B11
        
        if block == 0:
            a_block = 0; b_block = 0; c_block = 0
        elif block == 1:
            a_block = 1; b_block = 2; c_block = 0
        elif block == 2:
            a_block = 0; b_block = 1; c_block = 1
        else:
            a_block = 1; b_block = 3; c_block = 1
            
        for r in range(7):
            # U
            for i in range(2):
                for j_local in range(2):
                    j_global = a_block * 2 + j_local
                    U[i * 4 + j_global][col] = strassen_U[r][i][j_local]
            # V
            for j_local in range(2):
                for k_local in range(2):
                    j_global = (b_block // 2) * 2 + j_local
                    k_global = (b_block % 2) * 2 + k_local
                    V[j_global * 4 + k_global][col] = strassen_V[r][j_local][k_local]
            # W
            for k_local in range(2):
                for i in range(2):
                    k_global = c_block * 2 + k_local
                    W[k_global * 2 + i][col] = strassen_W[r][i][k_local]
            col += 1
            
    return {"U": U, "V": V, "W": W}
