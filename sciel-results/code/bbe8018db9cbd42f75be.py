import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=6
    # A is n x m (2x2)
    # B is m x p (2x6)
    # C is n x p (2x6)
    
    # We decompose into 3 blocks of 2x2x2 multiplication.
    # B = [B0, B1, B2] where Bi is 2x2.
    # C = [C0, C1, C2] where Ci is 2x2.
    # Ci = A Bi.
    
    # Rank of 2x2x2 multiplication is 7 (Strassen).
    # Total rank = 7 * 3 = 21.
    
    rank = 21
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen vectors for A (2x2)
    # A indices: 0:00, 1:01, 2:10, 3:11
    # U vectors are coefficients for linear combinations of A entries.
    # P1 = (A00 + A11)
    # P2 = (A10 + A11)
    # P3 = A00
    # P4 = A11
    # P5 = (A00 + A01)
    # P6 = (A10 - A00)
    # P7 = (A01 - A11)
    
    U_strassen = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    
    # For each block r in 0..2
    for r in range(3):
        base_col = r * 7
        # B block r corresponds to columns 2r and 2r+1 of B.
        # B indices: j*p + k. j in {0,1}, k in {2r, 2r+1}.
        # C block r corresponds to columns 2r and 2r+1 of C.
        # C indices: k*n + i. k in {2r, 2r+1}, i in {0,1}.
        
        for s in range(7):
            col = base_col + s
            
            # Fill U
            U[:, col] = U_strassen[s]
            
            # Fill V and W
            # V corresponds to B entries.
            # W corresponds to C entries.
            
            if s == 0: # P1: (A00+A11)(B00+B11)
                # V: B00 (j=0, k=2r) + B11 (j=1, k=2r+1)
                V[0 * p + 2*r][col] = 1
                V[1 * p + 2*r + 1][col] = 1
                # W: C00 (k=2r, i=0) + C11 (k=2r+1, i=1)
                # W index: k*n + i
                W[2*r * n + 0][col] = 1
                W[(2*r + 1) * n + 1][col] = 1
                
            elif s == 1: # P2: (A10+A11)B00
                # V: B00
                V[0 * p + 2*r][col] = 1
                # W: C10 (k=2r, i=1) - C11 (k=2r+1, i=1)
                W[2*r * n + 1][col] = 1
                W[(2*r + 1) * n + 1][col] = -1
                
            elif s == 2: # P3: A00(B01-B11)
                # V: B01 (j=0, k=2r+1) - B11 (j=1, k=2r+1)
                V[0 * p + 2*r + 1][col] = 1
                V[1 * p + 2*r + 1][col] = -1
                # W: C01 (k=2r+1, i=0) + C11 (k=2r+1, i=1)
                W[(2*r + 1) * n + 0][col] = 1
                W[(2*r + 1) * n + 1][col] = 1
                
            elif s == 3: # P4: A11(B10-B00)
                # V: B10 (j=1, k=2r) - B00 (j=0, k=2r)
                V[1 * p + 2*r][col] = 1
                V[0 * p + 2*r][col] = -1
                # W: C00 (k=2r, i=0) + C10 (k=2r, i=1)
                W[2*r * n + 0][col] = 1
                W[2*r * n + 1][col] = 1
                
            elif s == 4: # P5: (A00+A01)B11
                # V: B11
                V[1 * p + 2*r + 1][col] = 1
                # W: -C00 (k=2r, i=0) + C01 (k=2r+1, i=0)
                W[2*r * n + 0][col] = -1
                W[(2*r + 1) * n + 0][col] = 1
                
            elif s == 5: # P6: (A10-A00)(B00+B01)
                # V: B00 + B01
                V[0 * p + 2*r][col] = 1
                V[0 * p + 2*r + 1][col] = 1
                # W: C11 (k=2r+1, i=1)
                W[(2*r + 1) * n + 1][col] = 1
                
            elif s == 6: # P7: (A01-A11)(B10+B11)
                # V: B10 + B11
                V[1 * p + 2*r][col] = 1
                V[1 * p + 2*r + 1][col] = 1
                # W: C00 (k=2r, i=0)
                W[2*r * n + 0][col] = 1

    return {"U": U, "V": V, "W": W}
