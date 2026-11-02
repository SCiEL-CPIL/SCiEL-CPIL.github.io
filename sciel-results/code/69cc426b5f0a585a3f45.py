import numpy as np

def construct():
    n, m, p = 2, 2, 7
    rank = 25
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(offset_u, offset_v, offset_w, k_start):
        """
        Adds a Strassen decomposition for a 2x2 block of the output.
        offset_u: column offset in U, V, W
        k_start: starting k index for this block (0, 2, 4)
        """
        # Strassen terms for 2x2 matrix multiplication
        # C = A * B
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # U factors (rows 0..3 correspond to a,b,c,d)
        # V factors (rows 0..3 correspond to e,f,g,h)
        # W factors (rows 0..3 correspond to C00,C10,C01,C11)
        
        terms = [
            # (M1, M2, M3, M4, M5, M6, M7)
            # U coefficients for a,b,c,d
            [[1, 0, 1, 0, 1, 1, 0],   # a
             [0, 0, 0, 0, 1, 0, 1],   # b
             [0, 1, 0, 0, 0, 1, 0],   # c
             [1, 1, 0, 1, 0, 0, -1]], # d
            # V coefficients for e,f,g,h
            [[1, 1, 0, 0, 0, 1, 0],   # e
             [0, 0, 1, 0, 0, 1, 0],   # f
             [0, 0, 0, 1, 0, 0, 1],   # g
             [1, 0, -1, -1, 1, 0, 1]],# h
            # W coefficients for C00,C10,C01,C11
            [[1, 0, 0, 0, 0, 0, 0],   # C00 gets M1
             [0, 1, 0, 0, 0, 0, 0],   # C10 gets M2
             [0, 0, 1, 0, 0, 0, 0],   # C01 gets M3
             [0, 0, 0, 1, 0, 0, 0]],  # C11 gets M4
             # Wait, the standard Strassen output mapping is:
             # C00 = M1 + M4 - M5 - M7
             # C10 = M2 + M3
             # C01 = M5 + M6
             # C11 = M1 - M2 + M3 + M4
             
             # Let's re-verify the standard Strassen mapping.
             # M1 = (a+d)(e+h)
             # M2 = (c+d)e
             # M3 = a(f-h)
             # M4 = d(g-e)
             # M5 = (a+b)h
             # M6 = (c-a)(e+f)
             # M7 = (b-d)(g+h)
             
             # C00 = M1 + M4 - M5 - M7
             # C10 = M2 + M3
             # C01 = M5 + M6
             # C11 = M1 - M2 + M3 + M4
             
            # Corrected W coefficients
            [[1, 0, 0, 1, -1, 0, -1], # C00
             [0, 1, 1, 0, 0, 0, 0],   # C10
             [0, 0, 0, 0, 1, 1, 0],   # C01
             [1, -1, 1, 1, 0, 0, 0]]  # C11
        ]
        
        for t in range(7):
            col = offset_u + t
            # Fill U
            for i in range(2):
                for j in range(2):
                    idx = i * 2 + j
                    U[idx][col] = terms[0][i][j][t]
            # Fill V
            for j in range(2):
                for k in range(2):
                    idx = j * p + (k_start + k)
                    V[idx][col] = terms[1][j][k][t]
            # Fill W
            for k in range(2):
                for i in range(2):
                    idx = (k_start + k) * n + i
                    W[idx][col] = terms[2][k][i][t]

    def add_schoolbook_1col(offset_u, k_val):
        """
        Adds a schoolbook decomposition for a 2x2x1 block (1 output column).
        offset_u: column offset in U, V, W
        k_val: the k index for this block
        """
        # Schoolbook for C[:, k] = A * B[:, k]
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Each (i, j) pair contributes to W[i, k]
        for i in range(2):
            for j in range(2):
                col = offset_u
                U[i * 2 + j][col] = 1
                V[j * p + k_val][col] = 1
                W[k_val * n + i][col] = 1
                offset_u += 1

    # Block 1: k=0,1 (Strassen)
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: k=2,3 (Strassen)
    add_strassen_block(7, 0, 0, 2)
    
    # Block 3: k=4,5,6 (Strassen for k=4,5 + Schoolbook for k=6)
    # Sub-block 3a: k=4,5 (Strassen)
    add_strassen_block(14, 0, 0, 4)
    
    # Sub-block 3b: k=6 (Schoolbook)
    add_schoolbook_1col(21, 6)
    
    return {"U": U, "V": V, "W": W}
