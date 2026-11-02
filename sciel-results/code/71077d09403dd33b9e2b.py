    def add_strassen_block(A_r, A_c, B_r, B_c, C_r, C_c):
        nonlocal col
        # A indices
        A00 = A_r*3 + A_c
        A01 = A_r*3 + A_c + 1
        A10 = (A_r+1)*3 + A_c
        A11 = (A_r+1)*3 + A_c + 1
        
        # B indices
        B00 = B_r*8 + B_c
        B01 = B_r*8 + B_c + 1
        B10 = (B_r+1)*8 + B_c
        B11 = (B_r+1)*8 + B_c + 1
        
        # C indices (W)
        C00 = C_c*2 + C_r
        C01 = (C_c+1)*2 + C_r
        C10 = C_c*2 + C_r + 1
        C11 = (C_c+1)*2 + C_r + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        U[A00, col] = 1
        U[A11, col] = 1
        V[B00, col] = 1
        V[B11, col] = 1
        W[C00, col] = 1
        W[C11, col] = 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[A10, col] = 1
        U[A11, col] = 1
        V[B00, col] = 1
        W[C10, col] = 1
        col += 1
        
        # M3 = A00 (B01 - B11)
        U[A00, col] = 1
        V[B01, col] = 1
        V[B11, col] = -1
        W[C01, col] = 1
        W[C11, col] = 1
        col += 1
        
        # M4 = A11 (B10 - B00)
        U[A11, col] = 1
        V[B10, col] = 1
        V[B00, col] = -1
        W[C00, col] = 1
        W[C10, col] = 1
        col += 1
        
        # M5 = (A00 + A01) B11
        U[A00, col] = 1
        U[A01, col] = 1
        V[B11, col] = 1
        W[C00, col] = -1
        W[C01, col] = 1
        col += 1
        
        # M6 = (A10 - A00) (B00 + B01)
        U[A10, col] = 1
        U[A00, col] = -1
        V[B00, col] = 1
        V[B01, col] = 1
        W[C11, col] = 1
        col += 1
        
        # M7 = (A01 - A11) (B10 + B11)
        U[A01, col] = 1
        U[A11, col] = -1
        V[B10, col] = 1
        V[B11, col] = 1
        W[C00, col] = 1
        col += 1
