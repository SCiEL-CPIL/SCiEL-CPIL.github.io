import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    # Rank 28 decomposition
    U = np.zeros((n * m, 28), dtype=float)
    V = np.zeros((m * p, 28), dtype=float)
    W = np.zeros((p * n, 28), dtype=float)
    
    # Strassen terms for 2x2x2
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y11)
    # M4 = X11(Y01-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # Z00 = M7 + M5 - M2 + M4
    # Z01 = M1 + M7
    # Z10 = M3 + M5
    # Z11 = M7 - M1 + M3 + M6
    
    # Blocks:
    # 0: A_L (cols 0,1), B_LL (rows 0,1; cols 0,1) -> C_L (cols 0,1)
    # 1: A_R (cols 2,3), B_RL (rows 2,3; cols 0,1) -> C_L (cols 0,1)
    # 2: A_L (cols 0,1), B_LR (rows 0,1; cols 2,3) -> C_R (cols 2,3)
    # 3: A_R (cols 2,3), B_RR (rows 2,3; cols 2,3) -> C_R (cols 2,3)
    
    def add_strassen_block(block_idx, A_col_offset, B_row_offset, B_col_offset, C_col_offset):
        # A_col_offset: 0 or 2
        # B_row_offset: 0 or 2
        # B_col_offset: 0 or 2
        # C_col_offset: 0 or 2
        
        base_idx = block_idx * 7
        for s in range(7):
            r = base_idx + s
            
            # U part (A)
            # X is 2x2 from A. A[i, A_col_offset + j]
            # M1: X00, X11
            # M2: X10, X11
            # M3: X00
            # M4: X11
            # M5: X00, X01
            # M6: X10, X00
            # M7: X01, X11
            
            # V part (B)
            # Y is 2x2 from B. B[B_row_offset + j, B_col_offset + k]
            # M1: Y00, Y11
            # M2: Y00
            # M3: Y10, Y11
            # M4: Y01, Y00
            # M5: Y11
            # M6: Y00, Y01
            # M7: Y10, Y11
            
            # W part (C)
            # Z is 2x2 result. C[i, C_col_offset + k]
            # M1: Z01 (1)
            # M2: Z00 (-1)
            # M3: Z10 (1), Z11 (1)
            # M4: Z00 (1)
            # M5: Z00 (1), Z10 (1)
            # M6: Z11 (1)
            # M7: Z00 (1), Z01 (1), Z11 (1)
            
            # Helper to set U
            def set_u(i, j, val):
                idx_u = i * m + (A_col_offset + j)
                U[idx_u, r] += val
                
            # Helper to set V
            def set_v(j, k, val):
                idx_v = (B_row_offset + j) * p + (B_col_offset + k)
                V[idx_v, r] += val
                
            # Helper to set W
            def set_w(i, k, val):
                idx_w = (C_col_offset + k) * n + i
                W[idx_w, r] += val

            if s == 0: # M1
                set_u(0, 0, 1); set_u(1, 1, 1)
                set_v(0, 0, 1); set_v(1, 1, 1)
                set_w(0, 1, 1)
            elif s == 1: # M2
                set_u(1, 0, 1); set_u(1, 1, 1)
                set_v(0, 0, 1)
                set_w(0, 0, -1)
            elif s == 2: # M3
                set_u(0, 0, 1)
                set_v(1, 0, 1); set_v(1, 1, -1)
                set_w(1, 0, 1); set_w(1, 1, 1)
            elif s == 3: # M4
                set_u(1, 1, 1)
                set_v(0, 1, 1); set_v(0, 0, -1)
                set_w(0, 0, 1)
            elif s == 4: # M5
                set_u(0, 0, 1); set_u(0, 1, 1)
                set_v(1, 1, 1)
                set_w(0, 0, 1); set_w(1, 0, 1)
            elif s == 5: # M6
                set_u(1, 0, 1); set_u(0, 0, -1)
                set_v(0, 0, 1); set_v(0, 1, 1)
                set_w(1, 1, 1)
            elif s == 6: # M7
                set_u(0, 1, 1); set_u(1, 1, -1)
                set_v(1, 0, 1); set_v(1, 1, 1)
                set_w(0, 0, 1); set_w(0, 1, 1); set_w(1, 1, 1)

    add_strassen_block(0, 0, 0, 0, 0)
    add_strassen_block(1, 2, 2, 0, 0)
    add_strassen_block(2, 0, 0, 2, 2)
    add_strassen_block(3, 2, 2, 2, 2)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
