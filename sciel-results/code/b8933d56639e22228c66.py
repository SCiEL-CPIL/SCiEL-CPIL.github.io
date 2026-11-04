        def construct():
            n, m, p = 6, 6, 6
            # Strassen for 2x2 blocks of size 3x3
            # 7 multiplications
            # Each multiplication is 3x3 schoolbook (27 terms)
            # Total rank = 7 * 27 = 189
            
            rank = 189
            U = [[0]*rank for _ in range(36)]
            V = [[0]*rank for _ in range(36)]
            W = [[0]*rank for _ in range(36)]
            
            # Strassen definitions
            # A = [[A00, A01], [A10, A11]]
            # B = [[B00, B01], [B10, B11]]
            
            # M1 = A00 * (B00 + B11)
            # M2 = (A00 + A11) * B11
            # M3 = (A01 + A11) * B10
            # M4 = A00 * (B01 - B11)
            # M5 = (A10 - A00) * B00
            # M6 = (A00 + A01) * B10
            # M7 = (A10 - A01) * B01
            
            # Reconstruction
            # C00 = M2 + M3 - M5 - M7
            # C01 = M1 + M4
            # C10 = M5 + M6
            # C11 = M1 - M3 + M2 + M7
            
            # We need to map these to the global indices.
            # A00 is indices 0-8 of A
            # A01 is indices 9-17
            # A10 is indices 18-26
            # A11 is indices 27-35
            
            # For each Strassen block s (0..6):
            #   For each schoolbook term t (0..26):
            #       col = s * 27 + t
            #       Determine which A block and index is used in the left operand
            #       Determine which B block and index is used in the right operand
            #       Determine the contribution to which C block(s)
            
            pass
        