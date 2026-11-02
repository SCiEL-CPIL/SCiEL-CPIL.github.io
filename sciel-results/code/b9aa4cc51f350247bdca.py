import numpy as np

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x2 matrix by a 2x10 matrix.
    
    The rank is 35, achieved by applying the rank-7 Strassen algorithm 
    to each of the 5 independent 2x2 blocks of the multiplication.
    """
    n, m, p = 2, 2, 10
    rank = 35
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for A (2x2)
    # Rows correspond to a, b, c, d
    # Columns correspond to M1...M7
    strassen_A = [
        [1, 0, 1, 0, 1, -1, 0],  # M1, M2, M3, M4, M5, M6, M7
        [0, 0, 0, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 0]
    ]
    
    # Strassen coefficients for B (2x2)
    # Rows correspond to e, f, g, h
    strassen_B = [
        [1, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 1],
        [0, 0, -1, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0]
    ]
    
    # Strassen coefficients for C (2x2)
    # Rows correspond to C00, C01, C10, C11
    # Note: W index convention is W[k*n+i, r] where k is col of C, i is row of C.
    # C is p x n = 10 x 2.
    # For block b (0..4), the columns of C are 2*b and 2*b+1.
    # Row index in W: k*n + i.
    # For block b, k=2*b or 2*b+1. i=0 or 1.
    # So for block b:
    # C00 is at row (2*b)*2 + 0 = 4b
    # C01 is at row (2*b+1)*2 + 0 = 4b + 2  (Wait, C01 is row 0, col 1)
    # C10 is at row (2*b)*2 + 1 = 4b + 1  (Wait, C10 is row 1, col 0)
    # C11 is at row (2*b+1)*2 + 1 = 4b + 3
    
    # Let's re-verify the index mapping for W.
    # W[k*n + i, r]
    # k is column index of C (0..9)
    # i is row index of C (0..1)
    # For block b:
    # C00: k=2b, i=0 -> index 2b*2 + 0 = 4b
    # C01: k=2b+1, i=0 -> index (2b+1)*2 + 0 = 4b + 2
    # C10: k=2b, i=1 -> index 2b*2 + 1 = 4b + 1
    # C11: k=2b+1, i=1 -> index (2b+1)*2 + 1 = 4b + 3
    
    strassen_C = [
        # M1, M2, M3, M4, M5, M6, M7
        [1, 0, 0, 1, -1, 0, -1],  # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 0, 1]    # C11
    ]
    
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            # Fill U
            U[0, col] = strassen_A[0][r]
            U[1, col] = strassen_A[1][r]
            U[2, col] = strassen_A[2][r]
            U[3, col] = strassen_A[3][r]
            
            # Fill V
            # B indices for block b:
            # e: row 0, col 2b -> index 0*10 + 2b = 2b
            # f: row 0, col 2b+1 -> index 0*10 + 2b+1 = 2b+1
            # g: row 1, col 2b -> index 1*10 + 2b = 10 + 2b
            # h: row 1, col 2b+1 -> index 1*10 + 2b+1 = 10 + 2b+1
            
            V[2*b, col] = strassen_B[0][r]
            V[2*b+1, col] = strassen_B[1][r]
            V[10 + 2*b, col] = strassen_B[2][r]
            V[10 + 2*b+1, col] = strassen_B[3][r]
            
            # Fill W
            W[4*b, col] = strassen_C[0][r]    # C00
            W[4*b+2, col] = strassen_C[1][r]  # C01
            W[4*b+1, col] = strassen_C[2][r]  # C10
            W[4*b+3, col] = strassen_C[3][r]  # C11
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
