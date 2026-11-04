import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = 2, 4, 7
    # Rank estimation:
    # 3 blocks of 2x4x2 (each 2x2x2 Strassen x2 = 14) -> 42
    # 1 block of 2x4x1 (each 2x2x1 Schoolbook x2 = 8) -> 8
    # Total Rank = 50
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(a_i0, a_i1, b_j0, b_j1, c_k0, c_k1):
        """
        Adds Strassen decomposition for A[2x2] * B[2x2] = C[2x2].
        a_i0, a_i1: global indices for A rows (i=0,1)
        b_j0, b_j1: global indices for B cols (j=0,1)
        c_k0, c_k1: global indices for C cols (k=0,1)
        Note: A is 2x2, so A indices are (i, j_sub). 
        In our global A (2x4), j_sub maps to global j.
        """
        nonlocal col
        # A indices: (i, j)
        # A00: (a_i0, b_j0) -- Wait, b_j0 is B index.
        # Let's clarify indices.
        # A is 2x2. Rows i=0,1. Cols j=0,1.
        # B is 2x2. Rows j=0,1. Cols k=0,1.
        # C is 2x2. Rows i=0,1. Cols k=0,1.
        
        # Global A index: i * m + j_global
        # Global B index: j_global * p + k
        # Global C index: k * n + i
        
        # We pass global indices for the corners.
        # A00 is at (a_i0, b_j0) ? No.
        # a_i0 is the row index in A? No, A is 2x2.
        # Let's pass the global indices of the elements.
        
        # A elements:
        # A00: global (i=0, j=j0) -> idx_a00 = 0*m + j0
        # A01: global (i=0, j=j1) -> idx_a01 = 0*m + j1
        # A10: global (i=1, j=j0) -> idx_a10 = 1*m + j0
        # A11: global (i=1, j=j1) -> idx_a11 = 1*m + j1
        
        # B elements:
        # B00: global (j=j0, k=k0) -> idx_b00 = j0*p + k0
        # B01: global (j=j0, k=k1) -> idx_b01 = j0*p + k1
        # B10: global (j=j1, k=k0) -> idx_b10 = j1*p + k0
        # B11: global (j=j1, k=k1) -> idx_b11 = j1*p + k1
        
        # C elements:
        # C00: global (k=k0, i=0) -> idx_c00 = k0*n + 0
        # C01: global (k=k1, i=0) -> idx_c01 = k1*n + 0
        # C10: global (k=k0, i=1) -> idx_c10 = k0*n + 1
        # C11: global (k=k1, i=1) -> idx_c11 = k1*n + 1
        
        idx_a00 = 0 * m + b_j0 # Wait, b_j0 is j index.
        # Correction:
        # The function args should be j0, j1 for A cols/B rows, and k0, k1 for B cols/C cols.
        # i is always 0,1.
        
        # Let's rename args to j0, j1, k0, k1
        pass

    # Corrected helper
    def add_strassen_222(j0, j1, k0, k1):
        nonlocal col
        # A indices
        ia00 = 0 * m + j0
        ia01 = 0 * m + j1
        ia10 = 1 * m + j0
        ia11 = 1 * m + j1
        
        # B indices
        ib00 = j0 * p + k0
        ib01 = j0 * p + k1
        ib10 = j1 * p + k0
        ib11 = j1 * p + k1
        
        # C indices
        ic00 = k0 * n + 0
        ic01 = k1 * n + 0
        ic10 = k0 * n + 1
        ic11 = k1 * n + 1
        
        # Strassen M1..M7
        # M1 = (A00+A11)(B00+B11)
        U[ia00][col] += 1
        U[ia11][col] += 1
        V[ib00][col] += 1
        V[ib11][col] += 1
        W[ic00][col] += 1
        W[ic11][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[ia10][col] += 1
        U[ia11][col] += 1
        V[ib00][col] += 1
        W[ic10][col] += 1
        W[ic11][col] -= 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[ia00][col] += 1
        V[ib01][col] += 1
        V[ib11][col] -= 1
        W[ic01][col] += 1
        W[ic11][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[ia11][col] += 1
        V[ib10][col] += 1
        V[ib00][col] -= 1
        W[ic00][col] += 1
        W[ic10][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[ia00][col] += 1
        U[ia01][col] += 1
        V[ib11][col] += 1
        W[ic00][col] -= 1
        W[ic01][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[ia10][col] += 1
        U[ia00][col] -= 1
        V[ib00][col] += 1
        V[ib01][col] += 1
        W[ic11][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[ia01][col] += 1
        U[ia11][col] -= 1
        V[ib10][col] += 1
        V[ib11][col] += 1
        W[ic00][col] += 1
        W[ic10][col] -= 1
        col += 1

    def add_schoolbook_221(j0, j1, k0):
        nonlocal col
        # A is 2x2 (rows 0,1; cols j0,j1)
        # B is 2x1 (rows j0,j1; col k0)
        # C is 2x1 (rows 0,1; col k0)
        
        ia00 = 0 * m + j0
        ia01 = 0 * m + j1
        ia10 = 1 * m + j0
        ia11 = 1 * m + j1
        
        ib00 = j0 * p + k0
        ib10 = j1 * p + k0
        
        ic00 = k0 * n + 0
        ic10 = k0 * n + 1
        
        # C0 = A00*B0 + A01*B1
        U[ia00][col] += 1
        V[ib00][col] += 1
        W[ic00][col] += 1
        col += 1
        
        U[ia01][col] += 1
        V[ib10][col] += 1
        W[ic00][col] += 1
        col += 1
        
        # C1 = A10*B0 + A11*B1
        U[ia10][col] += 1
        V[ib00][col] += 1
        W[ic10][col] += 1
        col += 1
        
        U[ia11][col] += 1
        V[ib10][col] += 1
        W[ic10][col] += 1
        col += 1

    # Block 1: k=0,1. j split 0,1 and 2,3.
    # A1 = A[:, 0:2], B1 = B[0:2, 0:2] -> j0=0, j1=1, k0=0, k1=1
    add_strassen_222(0, 1, 0, 1)
    # A2 = A[:, 2:4], B2 = B[2:4, 0:2] -> j0=2, j1=3, k0=0, k1=1
    add_strassen_222(2, 3, 0, 1)
    
    # Block 2: k=2,3.
    add_strassen_222(0, 1, 2, 3)
    add_strassen_222(2, 3, 2, 3)
    
    # Block 3: k=4,5.
    add_strassen_222(0, 1, 4, 5)
    add_strassen_222(2, 3, 4, 5)
    
    # Block 4: k=6.
    # A1 = A[:, 0:2], B1 = B[0:2, 6] -> j0=0, j1=1, k0=6
    add_schoolbook_221(0, 1, 6)
    # A2 = A[:, 2:4], B2 = B[2:4, 6] -> j0=2, j1=3, k0=6
    add_schoolbook_221(2, 3, 6)
    
    if col != rank:
        # Fallback or error
        pass
        
    return {"U": U, "V": V, "W": W}
