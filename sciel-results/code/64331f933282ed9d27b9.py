import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We will fill the 36 columns.
    # Structure:
    # Block 1 (j=0,1): 
    #   Sub-block 1a (k=0,1): Strassen 2x2x2 (7 cols)
    #   Sub-block 1b (k=2,3): Strassen 2x2x2 (7 cols)
    #   Sub-block 1c (k=4): Schoolbook 2x2x1 (4 cols)
    # Block 2 (j=2,3):
    #   Sub-block 2a (k=0,1): Strassen 2x2x2 (7 cols)
    #   Sub-block 2b (k=2,3): Strassen 2x2x2 (7 cols)
    #   Sub-block 2c (k=4): Schoolbook 2x2x1 (4 cols)
    
    # Helper to add Strassen terms for 2x2x2
    # i_offset: 0 or 2 (row offset in A)
    # j_offset: 0, 1, 2, or 3 (col offset in A / row offset in B)
    # k_offset: 0, 1, 2, 3, or 4 (col offset in B / row offset in W)
    # start_col: starting column index in U, V, W
    
    def add_strassen(i_off, j_off, k_off, start_col):
        # A indices: (i, j) -> i*m + j. i in {0,1}, j in {j_off, j_off+1}
        # B indices: (j, k) -> j*p + k. j in {j_off, j_off+1}, k in {k_off, k_off+1}
        # W indices: (k, i) -> k*n + i. k in {k_off, k_off+1}, i in {0,1}
        
        # Strassen formulas:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # W contributions:
        # W00 = M1 - M4 + M5 - M7
        # W01 = M3 + M4 + M7
        # W10 = M2 + M3 - M5 - M6
        # W11 = M1 - M2 + M6
        
        # A local indices:
        # 00 -> i=0, j=j_off
        # 01 -> i=0, j=j_off+1
        # 10 -> i=1, j=j_off
        # 11 -> i=1, j=j_off+1
        
        a00_idx = 0 * m + j_off
        a01_idx = 0 * m + (j_off + 1)
        a10_idx = 1 * m + j_off
        a11_idx = 1 * m + (j_off + 1)
        
        # B local indices:
        # 00 -> j=j_off, k=k_off
        # 01 -> j=j_off, k=k_off+1
        # 10 -> j=j_off+1, k=k_off
        # 11 -> j=j_off+1, k=k_off+1
        
        b00_idx = j_off * p + k_off
        b01_idx = j_off * p + (k_off + 1)
        b10_idx = (j_off + 1) * p + k_off
        b11_idx = (j_off + 1) * p + (k_off + 1)
        
        # W local indices:
        # 00 -> k=k_off, i=0
        # 01 -> k=k_off, i=1
        # 10 -> k=k_off+1, i=0
        # 11 -> k=k_off+1, i=1
        
        w00_idx = k_off * n + 0
        w01_idx = k_off * n + 1
        w10_idx = (k_off + 1) * n + 0
        w11_idx = (k_off + 1) * n + 1
        
        # M1
        col = start_col
        U[a00_idx, col] += 1
        U[a11_idx, col] += 1
        V[b00_idx, col] += 1
        V[b11_idx, col] += 1
        W[w00_idx, col] += 1
        W[w11_idx, col] += 1
        col += 1
        
        # M2
        U[a10_idx, col] += 1
        U[a11_idx, col] += 1
        V[b00_idx, col] += 1
        W[w10_idx, col] += 1
        W[w11_idx, col] += -1
        col += 1
        
        # M3
        U[a00_idx, col] += 1
        V[b10_idx, col] += 1
        V[b11_idx, col] += -1
        W[w01_idx, col] += 1
        W[w10_idx, col] += 1
        col += 1
        
        # M4
        U[a11_idx, col] += 1
        V[b01_idx, col] += 1
        V[b00_idx, col] += -1
        W[w00_idx, col] += -1
        W[w01_idx, col] += 1
        col += 1
        
        # M5
        U[a00_idx, col] += 1
        U[a01_idx, col] += 1
        V[b11_idx, col] += 1
        W[w00_idx, col] += 1
        W[w10_idx, col] += -1
        col += 1
        
        # M6
        U[a10_idx, col] += 1
        U[a00_idx, col] += -1
        V[b00_idx, col] += 1
        V[b01_idx, col] += 1
        W[w10_idx, col] += -1
        W[w11_idx, col] += 1
        col += 1
        
        # M7
        U[a01_idx, col] += 1
        U[a11_idx, col] += -1
        V[b10_idx, col] += 1
        V[b11_idx, col] += 1
        W[w00_idx, col] += -1
        W[w01_idx, col] += 1
        col += 1
        
        return col

    def add_schoolbook_2x2x1(i_off, j_off, k_off, start_col):
        # A is 2x2, B is 2x1 (column vector).
        # C is 2x1.
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        
        a00_idx = 0 * m + j_off
        a01_idx = 0 * m + (j_off + 1)
        a10_idx = 1 * m + j_off
        a11_idx = 1 * m + (j_off + 1)
        
        b0_idx = j_off * p + k_off
        b1_idx = (j_off + 1) * p + k_off
        
        w0_idx = k_off * n + 0
        w1_idx = k_off * n + 1
        
        col = start_col
        
        # Term 1: A00 * B0 -> C0
        U[a00_idx, col] += 1
        V[b0_idx, col] += 1
        W[w0_idx, col] += 1
        col += 1
        
        # Term 2: A01 * B1 -> C0
        U[a01_idx, col] += 1
        V[b1_idx, col] += 1
        W[w0_idx, col] += 1
        col += 1
        
        # Term 3: A10 * B0 -> C1
        U[a10_idx, col] += 1
        V[b0_idx, col] += 1
        W[w1_idx, col] += 1
        col += 1
        
        # Term 4: A11 * B1 -> C1
        U[a11_idx, col] += 1
        V[b1_idx, col] += 1
        W[w1_idx, col] += 1
        col += 1
        
        return col

    # Block 1: j_off = 0
    col = 0
    col = add_strassen(0, 0, 0, col) # k=0,1
    col = add_strassen(0, 0, 2, col) # k=2,3
    col = add_schoolbook_2x2x1(0, 0, 4, col) # k=4
    
    # Block 2: j_off = 2
    col = add_strassen(0, 2, 0, col) # k=0,1
    col = add_strassen(0, 2, 2, col) # k=2,3
    col = add_schoolbook_2x2x1(0, 2, 4, col) # k=4
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
