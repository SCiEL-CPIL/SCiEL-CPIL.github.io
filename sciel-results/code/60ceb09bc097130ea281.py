import numpy as np

def construct():
    # Strassen decomposition for 2x2x2
    # u, v, w are vectors of length 4 (indices 00, 01, 10, 11)
    strassen_u = [
        [1, 0, 0, -1],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0]
    ]
    strassen_v = [
        [-1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    strassen_w = [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [-1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 1, 0]
    ]
    
    # Schoolbook decomposition for 2x2x3
    # a: 2x2 input (4 indices)
    # b: 2x3 input (6 indices)
    # c: 2x3 output (6 indices)
    schoolbook_a = []
    schoolbook_b = []
    schoolbook_c = []
    for i in range(2):
        for j in range(2):
            for k in range(3):
                a = [0]*4
                b = [0]*6
                c = [0]*6
                a[i*2 + j] = 1
                b[j*3 + k] = 1
                c[i*3 + k] = 1
                schoolbook_a.append(a)
                schoolbook_b.append(b)
                schoolbook_c.append(c)
                
    rank = 7 * 12
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(24)]
    
    col = 0
    for t in range(7):
        for s in range(12):
            # Process U (4x4 input)
            for i in range(2):
                for j in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            val = strassen_u[t][i*2 + j] * schoolbook_a[s][i2*2 + j2]
                            if val != 0:
                                # Flat index for 4x4 matrix
                                row = i*2 + i2
                                col_idx = j*2 + j2
                                flat_idx = row*4 + col_idx
                                U[flat_idx][col] = val
                                
            # Process V (4x6 input)
            # strassen_v is for 2x2 part of B
            # schoolbook_b is for 2x3 part of B
            for i in range(2):
                for j in range(2): # j is col in 2x2
                    for i2 in range(2): # i2 is row in 2x3
                        for j2 in range(3): # j2 is col in 2x3
                            val = strassen_v[t][i*2 + j] * schoolbook_b[s][i2*3 + j2]
                            if val != 0:
                                # B is 4x6.
                                # strassen_v index (i,j) corresponds to block row i, block col j?
                                # Wait, B is 2x2 in Strassen.
                                # In Kronecker product M_2,2,2 (otimes) M_2,2,3
                                # Input B is (2x2) (otimes) (2x3) -> 4x6.
                                # The 2x2 part is the "outer" block structure.
                                # The 2x3 part is the "inner" block structure.
                                # So Row = i*2 + i2, Col = j*3 + j2?
                                # Let's check dimensions.
                                # Outer: 2x2 blocks. Inner: 2x3 blocks.
                                # Total Rows: 2*2=4. Total Cols: 2*3=6.
                                # Row index = i*2 + i2.
                                # Col index = j*3 + j2.
                                row = i*2 + i2
                                col_idx = j*3 + j2
                                flat_idx = row*6 + col_idx
                                V[flat_idx][col] = val
                                
            # Process W (4x6 output)
            # strassen_w is for 2x2 output
            # schoolbook_c is for 2x3 output
            for i in range(2):
                for j in range(2): # j is col in 2x2
                    for i2 in range(2): # i2 is row in 2x3
                        for j2 in range(3): # j2 is col in 2x3
                            val = strassen_w[t][i*2 + j] * schoolbook_c[s][i2*3 + j2]
                            if val != 0:
                                row = i*2 + i2
                                col_idx = j*3 + j2
                                flat_idx = row*6 + col_idx
                                W[flat_idx][col] = val
                                
            col += 1
            
    return {"U": U, "V": V, "W": W}
