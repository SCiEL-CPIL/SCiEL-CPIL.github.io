def add_strassen_decomp(U, V, W, a_indices, b_indices, c_indices, start_col):
    """
    Adds a Strassen decomposition for a 2x2x2 multiplication to the global matrices.
    a_indices: list of 4 global indices for A (row-major 2x2)
    b_indices: list of 4 global indices for B (row-major 2x2)
    c_indices: list of 4 global indices for C (row-major 2x2)
    start_col: starting column index in U, V, W
    """
    # Strassen M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices in local 2x2:
    # A: 00, 01, 10, 11 -> indices 0, 1, 2, 3 in a_indices
    # B: 00, 01, 10, 11 -> indices 0, 1, 2, 3 in b_indices
    # C: 00, 01, 10, 11 -> indices 0, 1, 2, 3 in c_indices
    
    # M1
    col = start_col
    U[a_indices[0]][col] += 1
    U[a_indices[2]][col] += 1 # A11 is index 2? No.
    # Local A indices:
    # 0: (0,0)
    # 1: (0,1)
    # 2: (1,0)
    # 3: (1,1)
    # So A00=0, A01=1, A10=2, A11=3.
    
    # Correcting indices:
    # M1 = (A00+A11)(B00+B11)
    U[a_indices[0]][col] += 1
    U[a_indices[3]][col] += 1
    V[b_indices[0]][col] += 1
    V[b_indices[3]][col] += 1
    W[c_indices[0]][col] += 1 # C00
    W[c_indices[3]][col] += 1 # C11
    
    # M2 = (A10+A11)B00
    col = start_col + 1
    U[a_indices[2]][col] += 1
    U[a_indices[3]][col] += 1
    V[b_indices[0]][col] += 1
    W[c_indices[2]][col] += 1 # C10
    W[c_indices[3]][col] += -1 # C11
    
    # M3 = A00(B10-B11)
    col = start_col + 2
    U[a_indices[0]][col] += 1
    V[b_indices[2]][col] += 1
    V[b_indices[3]][col] += -1
    W[c_indices[1]][col] += 1 # C01
    W[c_indices[3]][col] += 1 # C11
    
    # M4 = A11(B01-B00)
    col = start_col + 3
    U[a_indices[3]][col] += 1
    V[b_indices[1]][col] += 1
    V[b_indices[0]][col] += -1
    W[c_indices[0]][col] += 1 # C00
    W[c_indices[2]][col] += 1 # C10
    
    # M5 = (A00+A01)B11
    col = start_col + 4
    U[a_indices[0]][col] += 1
    U[a_indices[1]][col] += 1
    V[b_indices[3]][col] += 1
    W[c_indices[0]][col] += -1 # C00
    W[c_indices[1]][col] += 1 # C01
    
    # M6 = (A10-A00)(B00+B10)
    col = start_col + 5
    U[a_indices[2]][col] += 1
    U[a_indices[0]][col] += -1
    V[b_indices[0]][col] += 1
    V[b_indices[2]][col] += 1
    W[c_indices[3]][col] += 1 # C11
    
    # M7 = (A01-A11)(B01+B11)
    col = start_col + 6
    U[a_indices[1]][col] += 1
    U[a_indices[3]][col] += -1
    V[b_indices[1]][col] += 1
    V[b_indices[3]][col] += 1
    W[c_indices[0]][col] += 1 # C00
