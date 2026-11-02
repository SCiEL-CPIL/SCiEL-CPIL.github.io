def construct():
    n, m, p = 2, 2, 9
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col_offset = 0
    # 4 Strassen blocks (each handles 2 columns)
    for block_idx in range(4):
        k_start = block_idx * 2
        k1 = k_start
        k2 = k_start + 1
        
        # Strassen terms
        # M1 = (A00 + A11)(B0k1 + B1k2) ? No, standard Strassen:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11) B00
        # M3 = A00 (B01 - B11)
        # M4 = A11 (B10 - B00)
        # M5 = (A00 + A01) B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
        # Indices for B: j*p + k
        # Indices for C: k*n + i
        
        # Term 1
        U[0][col_offset] += 1; U[3][col_offset] += 1
        V[0*p + k1][col_offset] += 1; V[1*p + k2][col_offset] += 1
        W[k1*n + 0][col_offset] += 1
        W[k2*n + 1][col_offset] += 1
        col_offset += 1
        
        # Term 2
        U[2][col_offset] += 1; U[3][col_offset] += 1
        V[0*p + k1][col_offset] += 1
        W[k1*n + 1][col_offset] += 1
        col_offset += 1
        
        # Term 3
        U[0][col_offset] += 1
        V[0*p + k2][col_offset] += 1; V[1*p + k2][col_offset] -= 1
        W[k2*n + 0][col_offset] += 1
        col_offset += 1
        
        # Term 4
        U[3][col_offset] += 1
        V[1*p + k1][col_offset] += 1; V[0*p + k1][col_offset] -= 1
        W[k1*n + 1][col_offset] += 1
        col_offset += 1
        
        # Term 5
        U[0][col_offset] += 1; U[1][col_offset] += 1
        V[1*p + k2][col_offset] += 1
        W[k2*n + 0][col_offset] -= 1
        W[k1*n + 1][col_offset] -= 1 # Wait, C11 = M1 + M4 - M5 + M7. C12 = M3 + M5.
        # Let's re-verify Strassen formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # My W assignments above were guesses. I need to be precise.
